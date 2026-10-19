# Mode compositing « live » pour Meshroom — spécification

> Statut : proposition / prototype (branche `live-compositing`).
> Objectif : permettre à Meshroom d'effectuer des opérations d'image en direct,
> en mémoire, à la manière de Nuke ou Natron, sans écrire de cache image par image,
> et à terme de relire des scripts Nuke (`.nk`).

## 1. Constat sur l'existant

Meshroom exécute ses graphes en mode **push / batch** :

- un nœud calculable est découpé en `NodeChunk`s (`meshroom/core/node.py`) ;
- chaque chunk est le plus souvent exécuté dans un sous-processus (`meshroom_compute`) ;
- les résultats et les fichiers de statut sont écrits dans `cache/<NodeType>/<uid>/` ;
- le Viewer2D ne lit que des chemins de fichiers (`FloatImageViewer` de QtAliceVision),
  et ne sait pas choisir un calque EXR (pas de gestion des AOVs) ;
- `SequencePlayer.qml` est lié à la photogrammétrie (`_currentScene.viewpoints`).

Un logiciel de compositing fonctionne en mode **pull** : on demande « la frame *t* »
à un nœud, la demande remonte récursivement sa branche, les résultats restent en RAM.

Le modèle de graphe de Meshroom (`Graph`, `Node`, `Attribute`, `Edge`, sérialisation
`.mg`, undo/redo, plugins, GraphEditor horizontal) est conservé. On ajoute un **second
moteur d'évaluation**, sans toucher au moteur batch.

## 2. Principes

| Sujet | Décision |
|---|---|
| Moteur | Python + numpy ; OpenImageIO pour les entrées/sorties (EXR multi-calques) |
| Évaluation | Pull, frame par frame, résultats en RAM |
| Cache | LRU en RAM plafonné en octets, clé `(uid du nœud, frame)` |
| Invalidation | Par l'UID existant de Meshroom : changer un paramètre change l'UID du nœud et de tout son aval |
| Affichage | Pas de nœud Viewer : clic droit → **Evaluate** sur un nœud |
| Recalcul | Automatique après modification d'un paramètre en amont du nœud évalué |
| Rendu final | Le nœud Write est un nœud batch classique (chunks, ferme de rendu) |

## 3. Le moteur live

### 3.1 Type de nœud `desc.ImageOpNode`

Nouveau type `MrNodeType.IMAGEOP` :

- **pas de pins flow** (`flowInputs` / `flowOutput`) : en mode live, la seule dépendance
  est la donnée ; évaluer un nœud ne calcule que ce que ses entrées demandent ;
- **non calculable en batch** (`isComputableType == False`) : pas de chunks, pas de
  dossier de cache, ignoré par `Compute` / `Submit` ;
- une ou plusieurs **entrées image** nommées (`desc.Image`, ex. `B`, `A`, `mask`) et une
  **sortie image** (`desc.Image` dans `outputs`).

Méthodes du descripteur :

```python
class Grade(desc.ImageOpNode):
    inputs = [desc.Image(name="input", label="Input"), desc.FloatParam(name="gain", ...)]
    outputs = [desc.Image(name="output", label="Output")]

    def getInfo(self, ctx, inputsInfo):          # format, plage de frames (défaut : celle de la 1re entrée)
        ...
    def render(self, ctx, inputs) -> ImageBuffer: # le calcul d'une frame
        ...
```

`ctx` (`RenderContext`) donne accès à la frame, au nœud (`ctx.node`) et à ses valeurs de
paramètres (`ctx.value("gain")`). `inputs` est un dictionnaire `{nomEntrée: ImageBuffer | None}`.

Deux variantes :

- **entrées « format seulement »** (`desc.Image(..., infoOnly=True)`, ou `infoOnlyInputs(ctx)`
  quand cela dépend d'un paramètre) : seul leur `ImageInfo` (format, plage) est utilisé, via
  `ctx.inputInfo(nom)` ; leurs pixels ne sont pas calculés. Les générateurs (Constant,
  CheckerBoard, ColorWheel) ont une entrée « Format » de ce type ; les nœuds Draw (Shape, Ramp,
  Roto) ont un choix « Input Usage » : `background` (dessin par-dessus) ou `format` ;
- **nœuds temporels** (`timeBased = True` : TimeOffset, FrameHold, Retime) : l'évaluateur ne
  calcule pas leurs entrées à la frame courante, ils demandent les frames voulues avec
  `ctx.renderInput(nom, frame)`.

### 3.2 L'attribut `desc.Image`

Un `desc.Image` est un `desc.Flow` spécialisé : il ne porte pas de valeur, seules les
connexions sont sérialisées. Contrairement au flow, son UID **propage l'UID du nœud amont**,
ce qui donne l'invalidation du cache gratuitement. Il ne se connecte qu'à un autre `Image`.

### 3.3 `ImageBuffer`

- pixels en `float32`, organisés en **calques nommés** (`rgba`, `depth`, `diffuse`, …),
  chaque calque ayant ses canaux (`R`, `G`, `B`, `A`, `Z`, …) ;
- `format` (display window) et **bbox** (data window), en coordonnées Nuke (origine en bas à
  gauche, y vers le haut). La bbox peut dépasser le format (**overscan**) :
  - le Read garde la data window des EXR ;
  - le Transform déplace la bbox (une translation en pixels entiers ne rééchantillonne rien) ;
  - le Blur l'agrandit de son rayon ;
  - le Merge prend l'union des bbox (option « Set BBox To ») ;
  - le Write écrit la data window dans les EXR et recadre au format pour les autres formats ;
- métadonnées (dictionnaire) ; les images sont immuables et partagent les calques non modifiés.

### 3.4 Passthrough et disable

- `isIdentity(ctx)` : un nœud à l'état neutre (Grade aux valeurs par défaut, Blur de taille 0,
  Transform identité, Merge sans A, mix à 0…) renvoie son entrée telle quelle, sans calcul ni mémoire ;
- **disable** (attribut interne `disable`, touche **I**, menu) : le nœud passe sa première entrée
  image ; un nœud source désactivé (Read, Constant) sort du noir transparent ;
- les nœuds avec une entrée mask ont un **Mask Channel** (alpha, red, green, blue, luminance) et
  **Invert Mask**.

### 3.5 L'évaluateur

`meshroom.live.evaluator.Evaluator` :

- `evaluate(node, frame)` : calcule récursivement la branche amont, en réutilisant le cache ;
- cache LRU en octets (`ImageCache`), clé `(node._uid, frame)` ; la mémoire est comptée par
  tableau de pixels (une image partagée n'est comptée qu'une fois). Budget : un quart de la RAM,
  ou `MESHROOM_LIVE_CACHE_GB` ;
- les opérations lourdes travaillent par bandes de lignes en parallèle (`MESHROOM_LIVE_THREADS`)
  et vérifient l'annulation entre les bandes ;
- un **statut d'évaluation** par nœud et par frame : `NONE`, `RUNNING`, `SUCCESS`, `ERROR`,
  plus `UPSTREAM_ERROR` (le nœud n'a pas pu calculer car une entrée a échoué) ;
- **annulation** : chaque requête porte un jeton ; une modification de paramètre rend les
  requêtes en cours obsolètes, elles s'arrêtent entre deux nœuds ;
- en interface, le calcul tourne dans un thread de travail (numpy et OpenImageIO relâchent le GIL).

### 3.6 Le pont avec le monde batch

- **Batch → live** : par des connexions d'attributs normales. Un `Read` de compo peut
  prendre son chemin de fichier en sortie d'un nœud de photogrammétrie. Si ce nœud n'est pas
  calculé, l'évaluation échoue avec le message « dépendance batch non calculée ».
- **Live → batch** : le `Write` est un `desc.Node` classique (chunks sur les frames, ferme
  de rendu, `meshroom_batch`). Son `processChunk` appelle l'évaluateur en mode headless.
  Il garde ses pins flow pour s'enchaîner avec le reste d'un pipeline.

## 4. Interface

### 4.1 Le mode « compositing »

- clé `workspaceMode: "compositing"` dans le header du `.mg` ;
- l'interface affiche alors l'espace de travail compositing : panneau d'affichage en haut,
  timeline dessous, graphe en bas, propriétés à droite. Le graphe reste horizontal.

### 4.2 Evaluate

- clic droit sur un nœud → **Evaluate** (raccourci : `E`) ;
- seule la branche amont est calculée ; le résultat s'affiche dans le panneau d'affichage ;
- après une modification de paramètre dans la branche, le nœud évalué se recalcule
  automatiquement ; un bouton permet de figer l'image ;
- une modification hors de la branche (création, déplacement, sélection d'un nœud) ne déclenche rien ;
- rien de lourd ne tourne dans le fil de l'interface : le rendu et la conversion pour l'affichage
  ont chacun leur thread.

### 4.3 Contours de statut

| Cas | Contour |
|---|---|
| Nœud hors de toute évaluation | aucun |
| Calcul en cours (affiché si > ~150 ms) | orange |
| Calcul réussi | vert |
| Calcul échoué | rouge, message au survol et dans le panneau |
| Entrée en échec (erreur héritée) | rouge atténué |
| Nœud évalué | même couleur, contour plus épais (~3 px) |

Le statut est celui de la frame courante. Le contour est indépendant de la bordure de sélection.

### 4.4 Connexions

- entrées et sortie image au premier plan, liens épais ;
- connexions entre paramètres conservées (équivalent des expressions Nuke), liens fins ;
- pas de pins flow sur les nœuds de compo.

Gestes prévus (P2) : lâcher un lien sur le corps d'un nœud le branche sur la première
entrée image libre, création d'un nœud à la suite de la sélection, insertion sur un lien,
nœud Dot.

### 4.5 Panneau d'affichage

Écrit à neuf, orienté AOVs :

- liste des calques de l'image, boutons de canaux R / G / B / A / luminance ;
- exposition (f-stops), gamma, transformation d'affichage (display / view de la config OCIO, ou
  intégrées : sRGB, Rec.1886, Rec.709, Display P3, chacune en Standard ou Raw ; clic droit sur le
  menu : recharger la config) ;
- valeur float sous le curseur, en coordonnées Nuke, overscan compris ;
- bbox en pointillés quand elle diffère du format ;
- conversion pour l'affichage à la résolution visible (un pixel sur 2ⁿ quand on dézoome) ;
- jauge de la mémoire du cache, à droite, et corbeille pour vider le cache du graphe (l'image
  affichée reste à l'écran) ;
- l'image et les contours sont dessinés en coordonnées écran (pas d'élément mis à l'échelle,
  quel que soit le zoom) ; bouton pour masquer l'overscan (masqué par défaut, la bbox reste en
  pointillés) ;
- barres d'outils des nœuds (Roto, Tracker) horizontales, en bas du viewer, juste au-dessus de la
  timeline ;
- plus tard : contact sheet de tous les AOVs, comparaison A/B.

### 4.6 Raccourcis, édition et connexions

- dans l'espace compositing, souris sur le graphe : **E** évalue, **I** désactive / réactive,
  **&** connecte la sortie du premier nœud sélectionné aux entrées des autres (première entrée
  compatible libre) ;
- clic droit sur une entrée ou une sortie : « Copy », puis sur une autre : « Connect to … »
  (sortie → entrée, ou entrée → même source que l'entrée copiée) ;
- souris sur le viewer : **R**, **G**, **B**, **A**, **Y** (luminance) ; la même touche revient au RGB ;
- souris sur le viewer ou la timeline : **J** / **K** / **L** (lecture arrière, stop, lecture
  avant), **←** / **→** (pas précédent / suivant) ;
- œil sur le nœud et dans ses propriétés pour le disable ; temps de calcul du nœud (lui seul) sur
  le nœud et dans ses propriétés ;
- valeurs 2D : champs x / y ; une valeur « liée » (Size du Blur, Scale du Transform…) s'édite avec
  un seul curseur, le bouton « + » sépare x et y (choix enregistré sur le nœud ; des valeurs x et y
  différentes restent toujours en x / y) ;
- éditeur de courbes : agrandissable (poignée sous la courbe), champs x / y du point sélectionné ;
- gizmo du Transform sélectionné dans le viewer : centre (translate), bras horizontal (scale),
  bras vertical (rotate) ; chaque glissé est une seule étape d'annulation ;
- gizmo du Ramp : ses deux points (orientation et longueur du fondu), et le losange du milieu
  pour déplacer les deux ;
- édition en direct : les curseurs des nœuds de compositing (valeurs simples, valeurs 2D, couleurs)
  mettent l'image à jour pendant le glissé, une seule étape d'annulation par glissé ;
- couleurs RGBA : pastille, valeurs, curseurs par canal et **pipette** : clic sur un pixel ou
  rectangle glissé (moyenne) dans le viewer, Échap pour arrêter. La couleur est prise dans
  l'image qui entre dans le nœud (ce que veut un keyer), sinon dans l'image affichée ;
- Roto : outils dans la barre à gauche du viewer quand le nœud est sélectionné (sélection,
  Bézier, rectangle, ellipse) ; sélection par clic ou fenêtre (Maj : ajouter), points et
  tangentes déplaçables (Maj : tangente cassée), double-clic : point lisse / anguleux, Suppr.
  Une Bézier est fermée dès son 3e point. Auto Key (par défaut) : clé à la frame courante à
  chaque modification, interpolation entre les clés. Feather global, sorties over / replaceAlpha /
  addAlpha / cutOut. La liste des formes (visibilité, nom, feather, opacité, invert, clés, lien à
  une piste) est dans les propriétés. Pas de rotopaint ; Alt + glisser déplace la vue.

### 4.7 Nœuds

| Menu | Nœuds |
|---|---|
| Comp Image | Read, Write, Constant, CheckerBoard, ColorWheel |
| Comp Draw | Shape (cercle / ellipse, rectangle, carré, triangle, feather), Ramp, Roto |
| Comp Color | Grade, ColorCorrect, Saturation, HueShift, Exposure, Multiply, Add, Clamp, Invert, ColorLookup, ColorLookupHSV, OCIOColorSpace |
| Comp Channel | Shuffle, Copy |
| Comp Keyer | Keyer, ChromaKeyer, Keylight (garbage / core mattes, vues final / matte / status) |
| Comp Filter | Blur, Sharpen, Erode, Median, MotionBlur, ZDefocus |
| Comp Merge | Merge, Premult, Unpremult, Dot, Clone, Switch |
| Comp Transform | Transform, Reformat (option Crop), Crop, Mirror, CornerPin, Tracker, Stabilize, CameraTracker, LensDistortion |
| Comp 3D | Camera, ReadGeo, MergeGeo, Resection (voir 4.13) |
| Comp Time | TimeOffset, FrameHold, Retime, TimeWarp, OpticalFlow |

Couleur du corps des nœuds par famille (quand l'utilisateur n'en a pas choisi), le bandeau gardant
la couleur sombre par défaut : Image (Read…) et Draw `#7B82A8`, les autres selon la charte de Nuke
(Color `#7AA9FF`, Channel `#9E3C63`, Keyer `#6CC77B`, Filter `#CC804E`, Merge `#4B5EC6`,
Transform `#A57AAA`, Time ocre, Write rouge, Dot et Clone gris).

- **Clone** (menu du nœud > Clone) : un nœud dont le lien vers le nœud cloné est caché ; il paraît
  déconnecté mais donne le résultat du nœud cloné et de tout son graphe amont. Il prend la couleur
  du nœud cloné ; le lien caché apparaît en pointillés quand le Clone ou le nœud cloné est
  sélectionné. Copier-coller un Clone (sans le nœud cloné) ou cloner un Clone donne un Clone du même
  nœud.
- **Valeurs 2D et par canal** : Blur (size), Transform (scale), Shape (size) ont une seule valeur
  par défaut et une case « split » pour éditer x et y ; ColorCorrect (contrast, gamma, gain, offset)
  et Grade (blackpoint, whitepoint, lift, gain, multiply, offset, gamma) de même, « split » donnant
  un éditeur de couleur (r, g, b, a) ; les valeurs par canal des Grade de Nuke sont importées. La case est enregistrée avec le
  nœud ; des valeurs différentes sont toujours montrées séparées.
- **Shuffle** : « Input Layer » est une liste des layers de l'image entrante (un EXR multi-layer,
  après un premier rendu) ; on peut aussi taper un nom.
- **Retime** : plages d'entrée (first / last) et de sortie (first / last), chacune prise sur
  l'entrée sauf si sa case est cochée ; sans dernière frame de sortie, la longueur vient de la vitesse.
- **ColorLookup** (master, rouge, vert, bleu, alpha) et **ColorLookupHSV** (teinte, saturation,
  valeur) : éditeur de courbes (clic : ajouter un point, glisser, double-clic : supprimer),
  pipettes « Source Color » et « Target Color » puis « Add Point » ; « Match Source » fait
  apparaître l'entrée « Match Target » et le bouton « Match » (correspondance d'histogrammes).
- **TimeWarp** : courbe frame de sortie → frame d'entrée.
- **Tracker** : pistes posées et suivies dans le viewer (corrélation normalisée puis affinage
  sous-pixel). Pendant le suivi, le viewer avance avec lui et les clés apparaissent au fur et à
  mesure (une seule étape d'annulation). Les frames posées à la main sont des keyframes (vignettes en
  haut du viewer), la fenêtre de zoom (en haut à gauche) agrandit la piste sélectionnée, « Error »
  colore le trajet du rouge (mauvaise corrélation) au vert. Avec 4 pistes sélectionnées (dans l'ordre
  bas-gauche, bas-droite, haut-droite, haut-gauche), le bouton CornerPin crée un CornerPin piloté.
- **Roto** : jusqu'à 3 Trackers (entrées Tracker 1 à 3) ; chaque forme a une liste de liens
  (tracker, piste, forme entière ou point N) ; un point avec son propre lien suit sa piste au lieu
  des liens de la forme entière. Avec 2 points sélectionnés ou plus, une boîte de transformation
  (comme Nuke) : glisser dedans déplace la sélection, les poignées carrées la mettent à l'échelle
  (depuis le côté opposé ; Ctrl : depuis le centre ; Shift sur un coin : uniforme), la poignée
  ronde la tourne ; un clic dans une zone vide de la boîte désélectionne. Une forme est fermée
  (remplie) ou ouverte : une courbe tracée comme un trait de sa largeur (bouton du toolbar et de la
  liste des formes). Entrée « Vectors » (un OpticalFlow) : « Track Shapes Forward / Backward »
  déplace les points des formes le long du flot optique (clé à chaque frame), pour les formes dont
  le bouton « flow » est actif, ou toutes, sauf celles liées à des pistes. Le suivi est interactif :
  la forme bouge dans le viewer frame après frame (le viewer suit), le tout restant une seule étape
  d'annulation ; arrêté en route, il garde les clés déjà posées (`ActionContext.publish`).
- **OpticalFlow** : flot optique de l'entrée (comme VectorGenerator) dans les layers « forward » et
  « backward » (u, v en pixels), Lucas-Kanade pyramidal sur une copie réduite (« Resolution »).
  Retime et TimeWarp ont un filtre « motion » : les deux frames voisines sont déplacées le long du
  flot puis mélangées ; « Motion Vectors » choisit de calculer le flot à la volée (« compute », sans
  autre nœud) ou de le lire sur l'entrée « Vectors » (un OpticalFlow) (`meshroom/live/flow.py`).
- **Switch** : sort l'une de ses entrées (0 à 3) selon « Which », animable ; une entrée non
  connectée donne du noir transparent. Option « Use Expression » : une expression Python choisit
  l'entrée, par exemple `1 if node('BG').file else 0` (`frame`, `node(nom)` par label ou nom,
  `nuke.toNode(nom)['knob'].value()` pour les scripts Nuke, `math` ; sur plusieurs lignes, elle
  affecte `result`). Les paramètres lus par l'expression sont relevés avec la branche
  (`desc.ImageOpNode.planContext`) et font partie de la clé du cache du Switch et de l'aval
  (`meshroom/live/expressions.py`). Les entrées non choisies peuvent échouer (Read sans fichier).
- **Reformat, entrée Format** : « To Format » prend le format de cette image quand elle est
  connectée (import des box `{{SRC.width}}` / `{{SRC.height}}` de Nuke).
- **LensDistortion** : modèles de 3DEqualizer, compatibles avec les nœuds LD de 3DE pour Nuke
  (« 3DE4 Classic LD Model », « 3DE4 Radial - Standard, Degree 4 » sphérique, « 3DE4 Anamorphic -
  Standard, Degree 4 » et « Degree 6 » anamorphiques avec le pixel aspect, « 3DE4 Anamorphic -
  Rescaled, Degree 4 », « 3DE4 Radial - Fisheye, Degree 8 » qui demande la focale et le filmback) :
  mêmes paramètres et unités (filmback, focale et lens center offset en cm) ; undistort, distort,
  ou leur STMap. « Import 3DE Node » lit un nœud LD_3DE4 d'un script Nuke (.nk) : modèle, caméra,
  paramètres, direction, et les knobs animés (zoom) comme clés (`meshroom/live/lens.py`, formules
  d'AliceVision et du Lens Distortion Plugin Kit de 3DE). La formule du facteur Rescale du modèle
  « Rescaled » (rayon multiplié avant le polynôme) reste à vérifier avec 3DE.
- **Tracker, Import From** : brancher un CameraTracker (ou un autre Tracker) sur cette entrée, puis
  « Import Tracks » copie ses pistes dans le Tracker (ajoutées, ou remplaçant les siennes avec
  « Replace Tracks ») et coupe le lien : elles deviennent éditables.
- **CornerPin** : homographie entre 4 coins « From » et « To », ou pilotée par 4 pistes.
- **Stabilize** : entrée Tracker ; transform ajustée par moindres carrés : translate (1 piste),
  rotate, scale, translate rotate, translate rotate scale (2 pistes), perspective (homographie :
  8 valeurs, donc 4 pistes au moins) ; option Match-Move.
- **MotionBlur** : entrée Tracker ; l'image est moyennée le long du mouvement des pistes pendant
  l'obturateur (shutter, offset, nombre maximal d'échantillons, environ un par pixel de mouvement),
  multiplié par « Factor ». Avec un CameraTracker, seules les pistes connues sur tout l'obturateur
  comptent (une feature de l'autotrack ne vit que quelques frames) ; de même pour Stabilize. Modèles : translate (moyenne des pistes, copies décalées en parallèle),
  translate rotate scale (ajusté sur les pistes), vector field (chaque piste donne le mouvement
  autour d'elle : champ de vecteurs interpolé, flou par pixel).
- **ZDefocus** : profondeur de champ. La profondeur vient de l'entrée Depth (passe Z CG, gradient
  de Roto ou de Ramp) ou d'un layer de l'entrée ; Math : depth (distance, 0 = infini), far = 0
  (1 / distance), linear (gradient), direct (taille du flou). Focal Plane, Depth Of Field, Size,
  Maximum Size ; bokeh : disque, lames (nombre, rotation), image de l'entrée Filter, aspect ratio
  (anamorphique). « Focal Plane Setup » montre le point (rouge devant, vert net, bleu derrière).
  L'image est découpée en tranches de profondeur floutées (FFT, à résolution réduite pour les
  grands rayons) et composées de l'arrière vers l'avant (`meshroom/live/defocus.py`).
- **CameraTracker** : « Autotrack » détecte et suit des points sur toute la séquence (en tâche de
  fond, avec progression ; les pistes apparaissent au fur et à mesure et le viewer suit la frame) ;
  « Solve Camera » calcule la caméra et les points 3D (paire initiale par matrice essentielle,
  résection PnP en plusieurs passes, triangulation, bundle adjustment avec scipy, intermédiaire
  contre la dérive puis final) ; les frames dont la caméra ne peut être résolue sont interpolées
  entre les frames résolues (le résumé en donne le nombre) ; « Estimate Focal Length » cherche la
  focale (12 à 180 mm) qui reprojette le mieux les pistes et l'écrit dans « Focal Length » ;
  « Export USD » écrit la caméra animée et le nuage de points en .usda, aux chemins choisis avec
  les boutons dossier de « Camera File » / « Points File » (`.usda` ajouté sans extension). Sinon rien n'est écrit : le nœud reste
  live, et sa sortie se branche comme un Tracker 2D. « Preview » montre ce que ferait l'autotrack à
  la frame courante avec les paramètres actuels (features, zones de recherche, appariement à la
  frame suivante : vert au-dessus de Min Correlation, rouge perdu). Entrée « Mask » : les features
  ne sont détectées et suivies que dans ses zones blanches.

Les pistes enregistrent le format de l'image suivie (`"format": [w, h]`) : leurs positions sont
en pixels de cette image et sont mises à l'échelle du format de l'image qui les utilise (suivi sur
un proxy 1K, utilisé sur la plaque 4K : `tracking.scaleTracks`).

**Animation** : toute valeur d'un nœud de compositing (nombres, booléens, choix, valeurs 2D,
couleurs) peut être animée. Le losange à côté de son nom pose une clé à la frame courante (orange :
clé à cette frame, bleu : animée, clé ailleurs ; clic droit : supprimer l'animation). Une fois
animée, toute modification pose une clé à la frame courante. Les clés sont dans l'attribut interne
`animation` (JSON `{param: [[frame, valeur], ...]}`, `meshroom/live/animation.py`), interpolées
linéairement (nombres et composantes), tenues pour le reste. Le rendu lit les clés frame par frame ;
la valeur stockée d'un paramètre animé ne fait que montrer la frame courante et n'entre pas dans
l'uid du nœud (les clés y entrent). Un paramètre lié à un paramètre animé suit ses clés. Les clés
apparaissent sur la timeline.

Un nœud de compositing enregistré avec une ancienne version de son descripteur est mis à jour
automatiquement au chargement (il ne porte aucune donnée calculée).

### 4.8 Read : sélecteur de séquences

Bouton dossier à côté du champ File du Read : navigateur qui regroupe les frames
(`render.####.exr  1001-1100 (100)`), filtre « Images Only ».

Le Read affiche sa plage (in-out) dans ses paramètres, à côté du champ File. « Frame Offset » décale la séquence (+10 : la
frame 1001 du fichier est montrée en 1011) ; « Outside Range » : tenir la première / dernière
frame, ou noir.

### 4.9 Timeline

Règle graduée avec les numéros de frame (façon Nuke), frame courante, frames en cache (vert), clés
du nœud sélectionné (jaune : Tracker, Roto, TimeWarp, paramètres animés) ; lecture avant / arrière, pas (step) en
frames, fps maximal, boucle, première / dernière frame. Pendant la lecture, les frames suivantes
sont calculées à l'avance ; une frame n'est montrée qu'une fois calculée et affichée.

Dans le viewer, un clic sur « Exp » ou « Gamma » les remet à 0 / 1.

L'éditeur de courbes (ColorLookup, TimeWarp) zoome avec Ctrl + molette (Shift : horizontalement,
Alt : verticalement), se déplace au bouton du milieu ou Alt + glisser, « fit » recadre ; le bouton
« ouvrir dans une fenêtre » le montre dans une fenêtre redimensionnable (plein écran possible), où
la molette seule zoome.

**Proxy** : le bouton « Proxy » du viewer (réduction 1/2, 1/3, 1/4 ou 1/8 dans le menu voisin ;
le titre du viewer l'indique) calcule à résolution réduite : les Reads
réduisent leurs images, les valeurs en pixels des nœuds suivent (attribut `proxyScaled` des
descripteurs, plus width / height ; formes Roto, pistes par leur format), et l'image est montrée à
la taille de la pleine résolution (gizmos et pipettes inchangés). Les images proxy ont leur propre
cache (`proxyKey`) ; le Write rend toujours en pleine résolution.

**ROI** (région d'intérêt, bouton « ROI » du viewer, actif par défaut) : quand le viewer est
zoomé sur une petite partie de l'image, seule la zone visible (élargie de la moitié de sa taille)
est calculée ; elle est recalculée quand la vue en sort. Les nœuds transmettent la région à leurs
entrées (`ImageOpNode.inputRegions`) : telle quelle pour les nœuds point à point (`pointwise` :
couleur, Merge, Shuffle, Copy, Roto...), élargie pour Blur, Erode, Median, Sharpen, ramenée par la
transformation inverse pour Transform ; les autres demandent leurs entrées entières. Une image
rendue pour une région n'est réutilisée que pour une région qu'elle couvre.

**In / out** : les bornes de la timeline sont enregistrées avec le graphe. Sans valeur entrée,
elles suivent le premier Read du graphe (celui créé quand il n'y en avait aucun ; les Reads créés
ensuite ne les changent pas) ; entrées à la main, elles restent (le bouton lien les fait suivre le
Read à nouveau) (`meshroom/live/frameRange.py`).

**Evaluate Compare** : le menu d'un nœud s'ouvre avec « Evaluate » sous le curseur ;
« Evaluate Compare » en fait la vue 2 : les touches 1 / 2 (pavé numérique aussi) passent de la vue 1
(Evaluate) à la vue 2, pour des avant / après.

**Graphe** : L range les nœuds sélectionnés (tout le graphe sans sélection) selon leurs
connexions, de gauche à droite : chaque nœud juste à droite de ses entrées, empilées dans l'ordre
de ses inputs (Merge : B en haut sur la ligne du nœud, A dessous, puis le mask) ; les edges cachés
et les backdrops ne comptent pas, une seule étape d'annulation (`meshroom/core/graphLayout.py`).
Un clic sur un edge le sélectionne (Shift : ajouter, Ctrl : basculer), un cadre au
bouton droit sélectionne les edges qu'il touche ; Suppr supprime les edges sélectionnés, H les
cache (ou cache les entrées des nœuds sélectionnés ; H à nouveau les montre) : un edge caché laisse
une pastille à chaque bout, dont l'info-bulle donne l'autre bout (sur une entrée de liste, retirer
un élément décale avec lui l'état caché des éléments suivants). Un clic sur un output tire un
edge qui suit le curseur : un clic sur une entrée le connecte, Tab crée un nœud connecté, Échap
l'annule. Un Clone affiche « ⇢ nœud cloné » puis son propre nom. Un cadre de sélection (bouton
gauche) qui n'attrape aucun nœud sélectionne les edges qu'il croise. Double-clic sur une pastille
d'edge caché : l'edge réapparaît ; clic droit : copier l'output ou l'input (puis « Connect » dans le
menu d'un pin), ou montrer l'edge. Créer un nœud (Tab) alors qu'un nœud de compositing est
sélectionné le place à sa droite, connecté à lui : des nœuds créés à la suite s'enchaînent.

Supprimer un nœud de compositing reconnecte sa source à ses sorties (comme Nuke).

### 4.10 Chapitres du graphe

Nœud `Chapter` (bouton « Add Chapter Line » de la barre du graphe) : une ligne horizontale
infinie, nommée (double-clic sur le titre) et colorée (propriété Color), pour découper un graphe
chargé en sections et montrer ses étapes à un nouvel arrivant.

- Une ligne suivie d'une autre ligne peut être **repliée** avec sa flèche (vers la droite :
  dépliée, vers le bas : repliée) : les nœuds entre les deux lignes disparaissent et tout ce qui
  est en dessous remonte, la ligne repliée se posant juste au-dessus de la suivante (l'écart
  comprend la place de la flèche) ; le titre indique le nombre de nœuds cachés.
- Le pliage ne change pas les positions enregistrées : c'est un décalage d'affichage
  (`meshroom/core/chapters.py`, `UIGraph.foldShift`). Il est enregistré dans le graphe et annulable.
- Déplacer une ligne ne déplace pas les nœuds de sa section.

### 4.11 Gestion de la couleur

Les images sont traitées dans un espace de travail linéaire (`meshroom/live/color.py`) :

- config OpenColorIO quand le module PyOpenColorIO est installé et que la variable d'environnement
  `OCIO` désigne une config : ses colorspaces, son rôle `scene_linear` comme espace de travail, ses
  displays / views pour le viewer ;
- sinon, colorspaces intégrés proches de ceux de Nuke : linear (primaires Rec.709, espace de
  travail), sRGB, rec709, Gamma1.8 / 2.2 / 2.4, Cineon, AlexaV3LogC, ACES2065-1, ACEScg, ACEScct,
  ACEScc, rec2020, Display P3, raw (courbes et primaires calculées, adaptation de Bradford) ;
- Read et Write : paramètre Colorspace (menu des colorspaces de la config, ou nom saisi) ; « auto »
  lit / écrit les fichiers entiers 8 / 16 bits en sRGB et les fichiers float dans l'espace de
  travail ; les courbes seules passent par les tables des fichiers entiers ;
- nœud OCIOColorSpace (In → Out, channels, mask, mix) ;
- import `.nk` : colorspaces des Read / Write (« ACES - ACEScg » → ACEScg, « default (...) » →
  auto), nœuds OCIOColorSpace et Colorspace.

### 4.12 Sous-graph de compositing par défaut

Passer en vue compositing (menu View > Compositing Workspace, File > New Compositing, ouverture d'un projet en
mode compositing) montre le compositing **dans un nœud SubGraph** : le premier SubGraph de compositing du graphe
(nommé ou labellisé `compositing...`, ou contenant des nœuds de compositing) est ouvert, ou un SubGraph
`compositing_1` est créé puis ouvert, avec le viewer et la timeline. Un projet qui a déjà des nœuds de compositing
dans son propre graphe (fait avant) reste tel quel. Le mode compositing et la plage de la timeline sont ceux du
projet (header du graphe racine), aussi à l'intérieur d'un sous-graph ; le viewer évalue les nœuds du graphe
affiché. Le template « compositing » contient `compositing_1` (Read → Grade) et le Write à l'extérieur
(`meshroom/core/subgraph/editing.py` : `compositingSubGraphs`, `UIGraph.ensureCompositingSubGraph`).

### 4.13 3D : flux caméra, géométrie et matrice

En plus des images, les nœuds de compositing échangent des **caméras**, des **géométries** et des **matrices de
transformation** (`desc.Camera`, `desc.Geo`, `desc.Matrix`, sous-classes de `desc.Stream` comme `desc.Image`) :
calculés frame par frame en mémoire par l'évaluateur (`ImageOpNode.computeData(ctx, sortie)`, lus avec
`ctx.inputData(entrée)`, cache par `(uid, sortie, frame)`), un flux ne se branche que sur un flux du même type.
Les edges de caméra sont ambre, de géométrie verts, de matrice violets (`meshroom/live/scene3d.py`).

- Conventions (USD, Nuke, Maya) : Y en haut, matrices 4x4 sur des vecteurs colonnes, transform = translate @
  rotate @ scale, angles en degrés dans l'ordre de rotation (ZXY par défaut : Z d'abord) ; une caméra est sa
  matrice caméra → monde et regarde vers -Z, l'ouverture horizontale (mm) couvre la largeur de l'image.
- **Sortie matrice de tous les nœuds** : chaque nœud de compositing a une sortie « Matrix », non exposée : son pin
  n'apparaît que quand le nœud est déplié (flèche du bas) ou qu'elle est branchée. C'est la transformation que le
  nœud applique à la frame : Transform, CornerPin, Tracker (match-move / stabilize), Stabilize, Reformat, Mirror
  (la 3x3 en pixels, plongée en 4x4), Camera, ReadGeo, Resection ; identité pour les autres.
- **Entrées matrice** (non exposées aussi) : « Input Matrix » de Transform, CornerPin, Camera, ReadGeo remplace la
  transformation du nœud (une caméra suit l'animation d'une autre caméra, un Transform reprend le mouvement d'un
  Tracker...).
- **Camera** : paramètres animables (translate, rotate, ordre de rotation, focale, ouvertures, near / far), ou
  fichier (USD : `.usda` lu sans module, binaire avec `pxr` ; Alembic avec PyAlembic ; FBX avec le SDK FBX ; nom de
  la caméra, décalage de frames, fps), ou entrée « Input Camera » (prise telle quelle, ex. la caméra d'une Resection).
- **ReadGeo** : OBJ, PLY (ascii / binaire), USD, Alembic, FBX (mêmes conditions), échantillons de temps lus à chaque
  frame ; translate / rotate / scale / uniform scale ou « Input Matrix ». **MergeGeo** : jusqu'à 4 géométries.
- **Resection** (à la GeoTracker) : entrées plaque, géométrie, caméra (optique et pose de départ ; sans caméra :
  35 mm sur 36 mm, cadrant la géométrie). Sorties : **Camera** (en premier : la caméra résolue, la géométrie fixe),
  **Camera Matrix**, **Geo Matrix** (la transformation de la géométrie vue depuis la caméra d'entrée, à brancher
  dans l'« Input Matrix » d'un ReadGeo), et l'image (la plaque avec la géométrie en fil de fer, les points des pins
  reprojetés et leur écart). Dans le viewer (Resection sélectionnée) : « Add pin » puis clic sur la géométrie (lancer
  de rayon depuis la caméra résolue ; nuage de points : le point projeté le plus proche) ; glisser le pin à sa place
  dans la plaque, la pose suit en direct : 1 pin déplace la géométrie dans l'image, 2 pins la tournent et la mettent
  à l'échelle, 3 pins et plus résolvent la pose (Gauss-Newton amorti vers la pose de départ ; DLT à partir de 6). Les
  pins se suivent comme les pistes d'un Tracker (avant / arrière) ; la pose est résolue frame après frame depuis la
  keyframe la plus proche (chaque pose part de la précédente). La barre d'outils indique le nombre de pins de la
  frame et l'erreur de reprojection. Un pin compte entre sa première et sa dernière clé (un pin à une seule clé
  la tient).
- **Disconnect and Bake** (clic droit sur un edge) : l'edge est retiré et l'entrée garde ce qu'elle recevait. Un
  paramètre garde la valeur de sa source, et ses clés si elle est animée (pour un enfant, x d'une valeur 2D : les
  clés du groupe). Une entrée caméra ou matrice devient des clés des paramètres du nœud sur la plage de la timeline
  (`ImageOpNode.bakeStream` : Camera → translate / rotate et optique, ReadGeo → translate / rotate / scale,
  Transform → translate / rotate / scale / skew, CornerPin → coins « To ») ; une seule étape d'annulation
  (`meshroom/live/bake.py`).

Limites : pas de distorsion d'objectif dans la Resection, focale non estimée, géométrie supposée fixe pendant que
l'on pose les pins (les points des pins sont en coordonnées monde) ; lecture Alembic / FBX non testée ici (modules
absents).

## 5. Import `.nk` (P4, fait)

File > « Import Nuke Script... », ou glisser un `.nk` sur le graphe : les nœuds du script sont
ajoutés au graphe (une étape d'annulation), et sa plage (Root first / last frame) devient l'in / out
de la timeline s'il n'a pas été entré à la main (`meshroom/live/nuke.py`).

- lecture : tokenizer (accolades, guillemets, crochets), puis machine à pile : un nœud prend ses
  N entrées (« inputs N », « 1+1 » : image puis mask) dans les N dernières entrées de la pile, le
  sommet étant l'entrée 0 (Merge2 : 0 = B, 1 = A, 2 = mask) ; `push 0`, `set` / `push $`,
  `clone` (un Clone du nœud cloné), Groups (leur contenu n'est pas importé) ;
- conversion des knobs par classe (Read, Write, Merge2, Grade, ColorCorrect, Blur, Transform,
  Constant, Crop, Reformat, Shuffle, Copy, Premult, Unpremult, Dot, Mirror, TimeOffset, FrameHold,
  Retime, TimeWarp, CheckerBoard2, Ramp, Radial, Rectangle, CornerPin2D, Keyer, Keylight,
  VectorGenerator, ZDefocus2, LD_3DE4, Saturation, Multiply, Add, Clamp, Invert, EXPTool, Sharpen,
  Median, FilterErode, OCIOColorSpace, Colorspace, Switch avec `which` en nombre, clés ou
  expression `[python ...]`) ; les knobs animés deviennent des clés, `{{Blur1.size}}` une connexion
  d'attributs ; Roto, RotoPaint et Tracker4 sont créés sans leurs formes / pistes ;
- classes inconnues → nœud `NukeUnsupported` qui garde la classe et les knobs bruts et laisse
  passer l'image ; les Viewer, Backdrop et StickyNote sont ignorés ;
- positions : xpos / ypos transposés pour garder le graphe horizontal ;
- un rapport liste ce qui n'est pas (entièrement) importé.

Reste à faire : formes Roto et pistes Tracker4, contenu des Groups, expressions générales,
comparaison des EXR produits avec ceux de Nuke.

## 6. Feuille de route

| Phase | Contenu |
|---|---|
| **P0** | Mode compositing dans le header, espace de travail dédié |
| **P1** | `ImageOpNode`, `desc.Image`, `ImageBuffer`, évaluateur + cache RAM, Evaluate, contours, panneau d'affichage minimal ; nœuds Read, Constant, Grade, Merge, Transform, Blur, Write |
| P2 | Timeline, paramètres animés (keyframes, fait : interpolation linéaire), gestes de connexion, Dot |
| P3 | ROI (fait), proxy (fait), OCIO (fait), bbox (fait) |
| P4 | Import `.nk` (fait, voir 5) + tests de comparaison avec Nuke |
| P5 | GPU, animation des formes Roto, rotopaint, hôte OpenFX |

## 7. Fichiers du prototype

| Fichier | Rôle |
|---|---|
| `meshroom/core/desc/node.py` | `MrNodeType.IMAGEOP`, `desc.ImageOpNode` |
| `meshroom/core/desc/attribute.py`, `meshroom/core/attribute.py` | attribut `Image` |
| `meshroom/core/node.py` | `IMAGEOP` non calculable en batch |
| `meshroom/core/graph.py`, `meshroom/core/graphIO.py` | `workspaceMode` dans le header |
| `meshroom/live/` | `ImageBuffer`, `ImageCache`, `Evaluator`, entrées/sorties image |
| `meshroom/nodes/compositing/` | nœuds Read, Constant, Grade, Merge, Transform, Blur, Write |
| `meshroom/ui/live.py` | pont Qt : évaluation en thread, statuts, image affichée |
| `meshroom/ui/qml/Compositing/` | espace de travail, panneau d'affichage |
| `meshroom/live/roto.py` | rasterisation des formes Roto (Bézier, antialiasing exact en x, sur-échantillonnage en y), clés et liens aux pistes |
| `meshroom/live/tracking.py`, `meshroom/live/curves.py` | suivi 2D (pistes indépendantes de la résolution) ; courbes (cubique monotone) et correspondance de couleurs |
| `meshroom/live/animation.py` | clés des paramètres animés (lecture, interpolation, édition) |
| `meshroom/live/defocus.py` | profondeur de champ (tranches, noyaux de bokeh, convolution FFT) |
| `meshroom/live/flow.py` | flot optique (Lucas-Kanade pyramidal), interpolation de frames |
| `meshroom/live/lens.py` | modèles de distorsion de 3DE, import des nœuds LD de Nuke |
| `meshroom/live/nuke.py`, `meshroom/nodes/compositing/NukeNodes.py` | import des scripts Nuke |
| `meshroom/live/color.py`, `meshroom/live/display.py` | colorspaces (OCIO ou intégrés), conversion pour l'affichage du viewer |
| `meshroom/live/frameRange.py` | in / out de la timeline (enregistrés dans le header du graphe) |
| `meshroom/core/chapters.py`, `meshroom/nodes/general/Chapter.py` | chapitres repliables du graphe |
| `meshroom/ui/qml/GraphEditor/` | menu Evaluate, contours de statut, éditeurs (couleur, 2D, formes), lignes de chapitre |
| `meshroom/live/scene3d.py`, `meshroom/nodes/compositing/ThreeDNodes.py` | 3D : caméras, géométries, matrices, lecture des fichiers, résolution de pose ; nœuds Camera, ReadGeo, MergeGeo, Resection |
| `meshroom/live/bake.py` | Disconnect and Bake |
| `tests/test_liveCompositing.py`, `tests/test_compositing3D.py` | tests du moteur et des nœuds |

## Subgraphs

Compositing nodes can be put in a `SubGraph` node (Ctrl+G): it is evaluated live through its subgraph, see [subgraph.md](subgraph.md#compositing-nodes-in-a-subgraph).
