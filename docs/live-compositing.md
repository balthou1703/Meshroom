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
- éditeur de courbes : agrandissable (barre sous l'éditeur, hauteur gardée pour la session par attribut :
  `LiveScene.editorSize`), sélection de plusieurs points (clic, Maj + clic : ajouter, Ctrl + glisser dans le vide :
  cadre, Ctrl + A : tous, Échap : aucun), glisser un point sélectionné déplace toute la sélection (l'ordre en x est
  gardé), Suppr / bouton corbeille / double-clic : supprimer ; champs x / y des points sélectionnés : un nombre les
  fixe, « +100 », « -=5 », « *2 », « /2 » les décalent ou les mettent à l'échelle tous (ex. TimeWarp : tous les points
  puis x « +100 ») ; interpolation des points sélectionnés : lisse (pente), linéaire (coin, sans pente) ou constante
  (palier jusqu'au point suivant), stockée en troisième valeur du point (`curves.INTERPOLATIONS`) ;
- gizmo du Transform sélectionné dans le viewer : centre (translate), bras horizontal (scale),
  bras vertical (rotate) ; chaque glissé est une seule étape d'annulation ;
- gizmo du Ramp : ses deux points (orientation et longueur du fondu), et le losange du milieu
  pour déplacer les deux ;
- gizmo du Crop (`CropGizmo.qml`) : la boîte (extérieur assombri), sa taille et son rapport affichés, poignées sur
  les bords et les coins (Maj sur un coin : rapport gardé), losange du milieu pour la déplacer, pixels entiers ;
  boutons Full / 1:1 / 16:9 / 1.85 / 2.39 : ce rapport cadré dans le format (pleine largeur ou hauteur, centré) ;
  le Crop affiché lui-même avec « Reformat » : l'image est la boîte, les poignées sur ses bords ;
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
  colore le trajet du rouge (mauvaise corrélation) au vert. Trois boutons suppriment des clés des pistes
  sélectionnées (toutes sans sélection), en gardant celle de la frame courante : avant elle, après elle, ou les clés
  trouvées par le suivi en gardant les keyframes posées à la main (une piste sans keyframe, d'un autotrack, n'est pas
  touchée) ; sur une Resection, les poses du suivi de géométrie suivent la même règle (une étape d'annulation,
  `tracking.deleteKeys`, `geotrack.deletePoses`). Toutes les pistes suivies avancent ensemble, frame après
  frame (chaque frame rendue une fois, `tracking.trackMany`) ; chacune est cherchée là où la mène son dernier
  mouvement (vitesse constante), puis là où elle était, et s'arrête seule (corrélation trop faible, sa prochaine
  keyframe). Avec 4 pistes sélectionnées (dans l'ordre
  bas-gauche, bas-droite, haut-droite, haut-gauche), le bouton CornerPin crée un CornerPin piloté.
- **Roto** : jusqu'à 3 Trackers (entrées Tracker 1 à 3). Un lien prend toutes les pistes d'un Tracker
  ou certaines (liste à cocher ; « toutes » suit aussi les pistes ajoutées ensuite) et la transformation
  qui les amène de la frame du lien à la frame courante, ajustée aux moindres carrés comme le Stabilize :
  translate, rotate, scale, translate rotate, translate rotate scale, perspective (avec moins de pistes
  connues qu'il n'en faut à une frame, leur mouvement moyen). Les « Roto Links » (paramètre `links`, son éditeur dans les attributs du Roto : liens à ajouter, pistes à
  cocher, transformation au menu) déforment toutes les formes ; les liens d'une forme la déplacent, elle ou un
  de ses points (un point avec son propre lien le suit au lieu des liens de la forme entière) ; ceux de la
  forme d'abord, puis ceux du Roto, plusieurs liens d'une même cible l'un après l'autre. Créer ou modifier
  un lien garde les formes où elles sont à la frame courante (`roto.relinkShape`). Le viewer reçoit les
  transformations du moteur (`rotoLinkTransforms`) pour afficher et éditer les formes liées. Les anciens
  liens à une piste (« track ») restent lus comme une translation. Avec 2 points sélectionnés ou plus, une boîte de transformation
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
- **Choix des pistes** (Stabilize, MotionBlur, CornerPin, paramètre « Tracks ») : un bouton ouvre la liste
  des pistes du Tracker connecté, à cocher (« All tracks » : toutes, aussi celles ajoutées ensuite ; CornerPin :
  4 pistes, dans l'ordre des coins, numérotées dans la liste). La valeur reste les noms séparés par des virgules.
- **Stabilize** : entrée Tracker ; transform ajustée par moindres carrés : translate (1 piste),
  rotate, scale, translate rotate, translate rotate scale (2 pistes), perspective (homographie :
  8 valeurs, donc 4 pistes au moins) ; option Match-Move.
- **MotionBlur** : entrée Tracker ; l'image est moyennée le long du mouvement des pistes pendant
  l'obturateur (shutter, offset, nombre maximal d'échantillons, environ un par pixel de mouvement),
  multiplié par « Factor ». Avec un CameraTracker, seules les pistes connues sur tout l'obturateur
  comptent (une feature de l'autotrack ne vit que quelques frames) ; de même pour Stabilize. Modèles : translate (moyenne des pistes, copies décalées en parallèle),
  translate rotate scale (ajusté sur les pistes), vector field (chaque piste donne le mouvement
  autour d'elle : champ de vecteurs interpolé, flou par pixel). « Motion From » : tracker (ci-dessus) ou camera :
  l'entrée Camera (un Camera lisant une caméra USD / Alembic / FBX, la caméra d'une Resection...) déplace l'image
  comme la caméra la voit bouger pendant l'obturateur : une homographie par échantillon (`_cameraMotion`), avec sa
  rotation et sa focale (zoom), et sa translation pour une image à « Distance » (0 : à l'infini, translation
  ignorée) ; la caméra est interpolée entre les frames. Seule l'entrée de la source choisie a un pin (les entrées
  désactivées n'ont pas de pin, sauf connectées).
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
« ouvrir dans une fenêtre » le montre dans une fenêtre redimensionnable aux couleurs de Meshroom (plein écran
possible), où la molette seule zoome.

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
Read à nouveau) (`meshroom/live/frameRange.py`). La frame courante est aussi enregistrée à la sauvegarde
(`currentFrame` du header, pas dans un template) et retrouvée à l'ouverture du projet.

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

Supprimer un nœud de compositing reconnecte sa source à ses sorties (comme Nuke). À l'inverse, un nœud glissé avec
Ctrl maintenu sur un edge (surligné en orange quand il le traverse) s'y intercale au relâcher : la source va dans son
entrée « input » (sinon sa première entrée compatible, une libre d'abord : B pour un Merge), sa sortie « output »
(sinon la première compatible) va à la destination ; pas sur ses propres edges ni s'il ferait un cycle
(`meshroom/core/edgeInsertion.py`, `UIGraph.insertNodeOnEdge`, une étape d'annulation).

Disposition des panneaux (`Application.qml`) : View > « Attribute Editor Full Height » ancre l'éditeur
d'attributs à droite sur toute la hauteur (comme le Properties Bin de Nuke) au lieu de sous le viewer ; réglage
mémorisé (`UILayout/attributeEditorFullHeight`). Ctrl+B (View > « Maximize Panel Under Cursor », comme Houdini)
affiche seul le panneau sous la souris (viewer, graphe ou attributs ; attributs par défaut), Ctrl+B ou le bouton
de la barre restaure. Les panneaux sont reparentés (`parent:` lié) vers des emplacements vides, pas recréés.

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
affiché. Le template « compositing » ne contient que `compositing_1` : Read → Grade → Write **à l'intérieur**, et
le port de sortie du SubGraph est un `File`, le chemin écrit par le Write. Calculer le SubGraph exécute le Write
dans sa tâche (toutes ses frames ; pour découper en tâches, mettre le SubGraph en boucle de frames)
(`meshroom/core/subgraph/editing.py` : `compositingSubGraphs`, `UIGraph.ensureCompositingSubGraph`).

File > Import Compositing... ajoute au compositing affiché celui d'un projet (`.mg` / `.mgt`) : le contenu de son
SubGraph de compositing pour un projet en mode compositing, sinon tout son graphe, avec chapitres et backdrops,
sans ses GraphInput / GraphOutput (`editing.readCompositing`, `UIGraph.importCompositing`, annulable). Le champ
Template d'un SubGraph n'est utilisé que si le sous-graph n'est pas embarqué (« Embedded Subgraph » vide).

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
- **Affichage de la géométrie** (Resection, « Display », `meshroom/live/geoRender.py`) : shaded (surface éclairée
  depuis la caméra, avec ses arêtes caractéristiques), edges (silhouettes, arêtes vives au-delà de
  « Crease Angle », bords ouverts : la forme d'un maillage dense sans le remplir), wireframe (toutes les arêtes des
  faces ; par défaut), points, none. « Hide Hidden Lines » retire ce qui est derrière la surface (tampon de profondeur : chaque
  pixel prend le triangle le plus proche qui contient son centre, un triangle plus petit qu'un pixel prend le pixel
  de son centre ; sur le GPU avec « Use GPU »). Lignes antialiasées. Les faces d'une géométrie (triangles, arêtes,
  voisinage des arêtes) sont une `Topology` partagée par toutes ses frames (`scene3d.Topology`) : calculées une fois,
  pas à chaque frame (c'était la latence du défilement de la timeline sur un maillage dense).
- **Resection** (à la GeoTracker) : entrées plaque, géométrie, caméra (optique et pose de départ ; sans caméra :
  35 mm sur 36 mm, cadrant la géométrie ; « Center Geometry » recule la caméra de départ le long de son axe jusqu'à
  centrer toute la géométrie, utile quand une grande géométrie entoure la caméra). Sorties : **Camera** (en premier : la caméra résolue, la géométrie fixe),
  **Camera Matrix**, **Geo Matrix** (la transformation de la géométrie vue depuis la caméra d'entrée, à brancher
  dans l'« Input Matrix » d'un ReadGeo), et l'image (la plaque avec la géométrie en fil de fer, les points des pins
  reprojetés et leur écart). Dans le viewer (Resection sélectionnée) : « Add pin » puis clic sur la géométrie (lancer
  de rayon depuis la caméra résolue ; nuage de points : le point projeté le plus proche) ; glisser le pin à sa place
  dans la plaque, la pose suit en direct : 1 pin déplace la géométrie dans l'image, 2 pins la tournent et la mettent
  à l'échelle, 3 pins et plus résolvent la pose (Gauss-Newton amorti vers la pose de départ ; DLT à partir de 6). Les
  pins se suivent comme les pistes d'un Tracker (avant / arrière), tous ensemble : à chaque frame, la pose est
  résolue avec les pins trouvés, un pin loin de l'endroit où elle met son point (glissement, occultation) s'arrête
  là (à partir de 4 pins ; avec 4, celui qui s’écarte le plus de sa position prévue), et chaque pin est cherché à la frame suivante là où la pose, à vitesse constante, le met
  (à partir de 3 pins). Comme dans GeoTracker, un pin ne contraint que les frames où il a une clé (posée à la main
  ou trackée) : la pose y est résolue avec les pins clés, de frame clé en frame clé depuis la pose trackée la plus
  proche (chaque résolution part de la précédente) ; entre deux frames avec une pose, elle est interpolée, et tenue
  avant la première et après la dernière. Aux autres frames, un pin est affiché (atténué) là où la géométrie résolue
  met son point (`pinProjections`) : le déplacer y ajoute une clé, et le tracker part de là. Même règle pour les pins
  du FaceTracker. La barre d'outils indique le nombre de pins de la frame et l'erreur de reprojection.
  **Suivi de la géométrie** (`meshroom/live/geotrack.py`, boutons de la barre d'outils et du nœud) : les pins ne
  servent qu'à placer la géométrie sur une keyframe ; « Track Geometry » (arrière / avant, de la frame courante à la
  keyframe suivante) fait suivre la plaque à de nombreux points de sa surface (« Track Points », 150) : points
  répartis sur les faces, ceux vus par la caméra (devant, face à elle, non cachés : tampon de profondeur grossier) sur
  une zone texturée de l'image (score de coin de Shi-Tomasi) ; leur mouvement vers la frame suivante est mesuré par le
  flot optique dense de la plaque (« Track Geometry With » : flow) ou en cherchant le patch de chacun (patches) là où la
  pose à vitesse constante le met, puis la pose est résolue en écartant les points qui ne suivent pas (`robustPose`).
  « Anchor To Keyframe » recale chaque pose sur la keyframe de départ (ses patches cherchés près de l'endroit où la
  pose les met), ce qui limite la dérive ; quand l'aspect a trop changé, la frame courante devient l'ancre.
  « Align » trouve la bonne position à la frame courante depuis une position approximative (géométrie placée à peu
  près, à la main ou par les pins) contre la keyframe la plus proche : patches cherchés de plus en plus près (81, 41,
  21 pixels) ; la frame devient une keyframe. « Refine » refait le suivi entre chaque paire de keyframes dans les deux
  sens et mélange les deux poses selon la distance à chaque keyframe (la dérive répartie). Les poses suivies sont dans
  « Geometry Track » (JSON : poses et keyframes) et priment sur les pins, sauf aux keyframes des pins (qui les
  ajustent : la pose y part de la pose suivie) ; hors des frames suivies, la pose suivie la plus proche est tenue
  quand il n'y a pas de pin ; « Clear Geometry Track » les retire. Si le suivi s'arrête (trop peu de points visibles
  ou trouvés), les poses déjà trouvées sont gardées et la raison s'affiche. Entrée « Mask » (non rendue, lue par les
  actions ; canal et inversion comme les autres masques) : Track Geometry, Align et Refine ne suivent que les points
  dans sa partie blanche (`geotrack.setMask` / `maskedOut` : un objet qui passe devant ou une partie qui bouge, son
  roto inversé).
- **Deform With Flow** (Resection, non rigide, `geotrack.trackDeformation`) : depuis la frame courante jusqu'au bout
  de la plage de tracking (avant / arrière), des sommets de contrôle répartis sur la géométrie (« Deform Points »,
  400) suivent le flot optique de la plaque (Lucas-Kanade, ou RAFT avec « Track Geometry With » : raft) à partir de
  la pose rigide ; un sommet caché par la géométrie ou masqué garde son décalage. Leurs décalages (pixels, par
  rapport à la pose rigide) sont gardés dans « Geometry Deformation » ; chaque sommet suit ses contrôles voisins
  (gaussiennes normalisées, `DeformWeights`) dans l'image, à sa profondeur. Sortie « Geo » : la géométrie déformée
  (celle de l'entrée sans déformation), affichée par le viewer ; « Clear Deformation » la retire ; les boutons de
  suppression avant / arrière retirent aussi ses frames.
- **Disconnect and Bake** (clic droit sur un edge) : l'edge est retiré et l'entrée garde ce qu'elle recevait. Un
  paramètre garde la valeur de sa source, et ses clés si elle est animée (pour un enfant, x d'une valeur 2D : les
  clés du groupe). Une entrée caméra ou matrice devient des clés des paramètres du nœud sur la plage de la timeline
  (`ImageOpNode.bakeStream` : Camera → translate / rotate et optique, ReadGeo → translate / rotate / scale,
  Transform → translate / rotate / scale / skew, CornerPin → coins « To ») ; une seule étape d'annulation
  (`meshroom/live/bake.py`).

Limites : pas de distorsion d'objectif dans la Resection, focale non estimée, géométrie supposée fixe pendant que
l'on pose les pins (les points des pins sont en coordonnées monde) et rigide pendant son suivi (la géométrie de la
frame courante) ; suivi en numpy sur CPU (environ 0,1 à 0,3 s par frame à 1K, davantage en 4K avec le flot optique ;
voir 4.14 pour le GPU) ; lecture Alembic / FBX non testée ici (modules absents).

### 4.13 bis FaceTracker

**FaceTracker** (Comp 3D, `meshroom/nodes/compositing/FaceNodes.py`, `meshroom/live/face.py`) : un visage suivi dans
la plaque, bouche et yeux compris (non rigide). « Track Face » passe MediaPipe Face Landmarker (Google, Apache-2.0,
CPU) sur toute la plage : 478 points par frame et 52 coefficients d'expression, gardés dans le nœud (« Face Track »).
Le landmarker (qui travaille sur ~256 pixels) voit un carré autour du visage de la frame précédente, sinon toute
l'image, sinon des tuiles carrées de la hauteur de l'image puis de sa moitié : un petit visage dans une grande plaque
est trouvé ; chaque carré sombre est éclairci (au plus x8). Un visage très masqué (feuillage) peut rester introuvable.
Entrée « Mask » (canal et inversion comme les autres masques) : le visage n'est cherché que dans sa partie blanche
(le reste noirci, recherche limitée à sa boîte) ; les points qu'il masque (« h » dans « Face Track ») ne placent pas
la tête, leurs sommets suivent seulement la tête, et ils n'ont pas de clé de piste à ces frames (un feuillage devant
le visage : son roto inversé).
Pins (« Pins », comme GeoTracker, `face.solveFace(pins=...)`) : dans le viewer, « Add pin » puis clic sur le visage
(le point est gardé sur le visage neutre : triangle et coordonnées barycentriques), puis glisser le pin à sa place ;
les boutons Track suivent les pins comme des pistes. Ils placent la tête avec les points rigides (comptés trois fois),
puis, avec « Non Rigid », le visage est déplacé autour de chacun (gaussiennes sur le visage neutre de rayon « Pin
Radius », résolues pour que chaque pin passe exactement par sa position) : un pin au coin des lèvres tire la lèvre,
pas la tête. Le viewer les dessine reliés au point du visage (l'erreur) ; « Show Pin Reach » teinte le maillage en
orange là où les pins le déplacent (le rayon d'impact). « Track With » : mediapipe (MediaPipe seul à chaque frame),
flow (le visage ajusté de la frame courante, pins compris, suit le flot optique de la plaque : Lucas-Kanade ou RAFT,
« Optical Flow »), mediapipe+flow (le flot, et MediaPipe cherché autour du visage suivi, pris là où il s'accorde
avec lui : pas de dérive, et le flot là où MediaPipe se perd ; `face.propagateFaces`). Avec le flot, partir d'une
frame bien placée : boutons du viewer (et du nœud) en arrière / avant jusqu'au bout de la plage de tracking, ou une
frame à la fois (le viewer passe à la frame trackée) ; les autres frames gardent leurs points. Les pistes ont aussi
des boutons d'une frame (avant / arrière) dans la barre du viewer (Tracker, Resection).
Comme dans KeenTools FaceTracker, les pins du FaceTracker appartiennent à leurs keyframes : une frame où des pins
ont une clé est une keyframe (le visage y est calé avec ses pins) ; ailleurs ils sont cachés et ne contraignent rien
(teinte « Show Pin Reach » comprise), et ils ne sont pas trackés (pas de boutons de tracking de pins) : c'est le
visage qui est tracké, de la keyframe courante jusqu'à la suivante (exclue) ou le bout de la plage. « Refine »
retracke entre chaque paire de keyframes depuis les deux bouts et mélange les deux suivis, pondérés vers la keyframe
la plus proche.
« Stiffness » (non rigide, `face.regularizedShifts`) : le visage n'est pas amené sommet par sommet sur ses points
mais comme une surface lisse passant par eux (moindres carrés pondérés + laplacien des déplacements sur le maillage,
repondérés de façon robuste) : un point faux (dérive, détection ratée) est ignoré, les points masqués suivent leurs
voisins, le maillage ne se replie pas ; 0 : chaque sommet exactement sur son point. Pendant le suivi par le flot, le
visage ainsi ajusté est ce qui est suivi d'une frame à l'autre (il reste un visage).
« Non Rigid Model » : arkit (par défaut, `faceRig.BlendshapeRig`) : le visage est déformé par les 51 formes
d'expression d'ARKit (eyeBlinkLeft, jawOpen, mouthSmileRight… : les noms d'ARKit, de MediaPipe et des rigs faciaux de
Maya, Unreal Live Link, Blender ; tongueOut en moins), chacune pondérée de 0 à 1. Les formes viennent d'ICT-FaceKit
(MIT), transférées sur la tête du FaceTracker par `meshroom/live/data/buildFaceExpressions.py` →
`data/faceExpressions.npz` : tête ICT alignée (similitude sur les 68 points) puis déformée sur le visage MediaPipe
(thin plate spline sur les 68 points et les contours des paupières et des lèvres intérieures, appariés par abscisse
curviligne entre leurs coins) ; chaque sommet prend les déplacements de son point le plus proche sur la tête ICT
déformée, du même côté des fentes des yeux et de la bouche (un sommet de paupière haute suit la paupière haute) ;
lissage léger sur notre maillage, contours gardés ; les clignements sont mis à l'échelle pour fermer nos yeux (l'œil
canonique de MediaPipe est plus ouvert) ; les sommets de la tête sur l'ovale suivent leur arête. Côtés : Left = gauche
du visage (+X). Sur une autre forme de repos (l'acteur : FaceBuilder, Fit Shape), les déplacements sont mis à
l'échelle de la largeur du visage. Même ajustement que le rig ci-dessous (`fitRig`, généralisé : jacobien, limites
et coûts fournis par le rig) : poids bornés à [0, 1], formes surtout en profondeur (cheekPuff, pucker, roll) plus
coûteuses ; le contrôle de repliement ignore les triangles dégénérés et ceux qui traversent les yeux et la bouche
(fermés par blink / mouthClose), et seuls les triangles vraiment retournés atténuent l'expression.
« MediaPipe Expressions » (0,1 par défaut) : sur les frames trouvées par MediaPipe, ses scores de blendshapes
(mêmes noms) guident les formes en plus des points (`BlendshapeRig.fromMediaPipe` ; à 1, autant que les points) ;
gauche et droite de MediaPipe sont moyennées (son côté n'est pas fiable), les points les distinguent.
Suivi (flow → rig → shape) : pendant le suivi par le flot, c'est le visage du rig (sans la déformation locale des
pins : `report["rigWorld"]`) qui est suivi à la frame suivante ; la correction locale des pins reste propre à sa
frame et ne dérive pas.
rig (`meshroom/live/faceRig.py`, `FaceRig`) : comme le modèle de tête de KeenTools
FaceTracker, le visage est déformé par un rig construit sur la topologie MediaPipe (axes tirés du visage : menton →
front, un côté à l'autre) : la mâchoire (rotation ouverture / latérale de la partie sous la ligne des lèvres autour
d'une charnière près des oreilles), sourcils intérieur / extérieur, paupières haute / basse (la haute ne bouge pas
la basse), coins de la bouche, lèvre haute / basse, joues, bout du nez et ailes du nez : une translation 3D chacun, de sa zone (chute
gaussienne). Front, crâne, arête du nez, tempes restent rigides. L'ajustement (`fitRig`) résout ensemble la pose de
la tête et les ~53 paramètres (Levenberg-Marquardt, jacobien analytique, poids de Cauchy des points), régularisé vers
le repos (« Stiffness ») ; les pins déplacent d'abord les contrôles (comptés 20 fois), puis ce que le rig ne peut
pas faire est fait par une déformation locale autour de chaque pin (exacte au pin, sur « Pin Radius ») : les pins
sont atteints, comme dans KeenTools ; « Show Pin Reach » montre les zones des contrôles sous chaque pin et la
déformation locale.
Garde-fous : les points du côté du visage tourné à l'opposé de la caméra (profil : MediaPipe les devine) ne sont pas
utilisés ; les contrôles ont des limites douces (translation : un cinquième de la largeur du visage ; mâchoire :
-0.15 à 0.6 rad d'ouverture, ±0.25 latéral) ; leurs mouvements en profondeur (mal vus) coûtent trois fois plus.
surface : chaque sommet libre (surface lisse, ci-dessus).
Pins du FaceBuilder (comme KeenTools FaceBuilder) : les oreilles, le crâne et le cou, que les landmarks ne voient
pas, sont placés par des pins posés dans les vues (un paramètre par vue, `pins` … `pins3` ; gizmo du viewer sur la vue
affichée, « Show View ») : un pin est un point de la tête de base (clic sur la tête vue par la caméra de la vue) et sa
position dans la photo (sa clé à la frame de la vue ; une photo : la clé la plus proche). `face.warpToPins` : autour
de chaque pin la surface bouge en douceur sur « Pin Radius » (gaussienne de la distance 3D : indépendante de la
densité du maillage, locale : un pin d'oreille ne déplace ni le cou ni l'autre côté), les pins résolus ensemble
(Gauss-Newton sur leur reprojection, amorti : un pin vu dans une seule vue bouge dans cette vue, des pins proches dans
plusieurs vues placent la surface en 3D), chacun exact. Un pin de la tête (oreilles, crâne, cou) ne bouge pas le visage
(ses landmarks l'ont placé), sauf près de l'ovale où le mouvement s'estompe sur 2 cm ; un pin posé sur le visage le
bouge. Quelques millisecondes : la tête suit le pin pendant qu'on le glisse.
« Fit Ears, Neck And Skull » (FaceBuilder, `meshroom/live/headFit.py`) : après « Build Head », les oreilles, le cou
et le crâne ajustés tout seuls sur les vues. Chaque vue (recadrée autour de la tête) est segmentée par le segmenteur
multiclasse de MediaPipe (`selfie_multiclass_256x256` : cheveux, peau du visage, peau du corps, vêtements…) ; les
sommets du contour extérieur de la tête (entre faces vues et cachées, rien de la tête au-delà) des oreilles et du cou
sont amenés, le long de la normale du contour dans l'image, sur le bord de la peau le plus proche ; ceux du crâne,
cachés par les cheveux, seulement ramenés dans la tête (cheveux ou peau) quand ils en sortent. Avec « Use Depth », la
profondeur relative de Depth Anything V2 Small (transformers) est calée sur le visage visible (1 / profondeur = a ×
disparité + b, robuste ; ignorée si elle ne suit pas le visage) et place en profondeur les oreilles et le cou vus sur
la peau (pas sur un bord de profondeur) ; son poids total vaut la moitié de celui des contours. Le déplacement : des
gaussiennes (« Fit Radius ») autour de points de contrôle répartis sur la tête (échantillonnage le plus éloigné), le
visage gardé (sauf près de l'ovale, `face.headMobility`), contraintes robustes (Cauchy), toutes les vues ensemble,
recalculé depuis les nouveaux contours (5 itérations), les déplacements régularisés. Stocké dans « Head Fit » (points
de contrôle et leurs déplacements), appliqué avant les pins. Modèles : `python -m meshroom.live.neural.download
headsegmenter depthanything` (Apache-2.0). Sur des vues synthétiques : erreur des oreilles, du cou et du crâne
divisée par ~2 avec les contours, par ~2,5 avec la profondeur.
« Head From Face » (FaceBuilder, activé par défaut) : la tête entière prend les proportions qui vont avec le visage
ajusté, via le modèle d'identité d'ICT-FaceKit (MIT ; 50 modes PCA appris sur des scans de têtes entières, transférés
sur notre topologie par `meshroom/live/data/buildFaceExpressions.py` → `faceIdentity.npz`). Les coefficients sont
prédits par moindres carrés régularisés sur le visage (`headFit.predictIdentity`) ; la tête est
`identityHead` = repos + modes·c, corrigée pour que le visage soit exactement celui ajusté. « Fit Ears, Neck And
Skull » n'ajuste alors plus des déplacements libres mais ces ~50 coefficients (`headFit.fitIdentity`, Gauss-Newton
sur les mêmes contraintes de contour et de profondeur, ramené vers la prédiction) : le résultat reste une tête humaine
même quand la segmentation se trompe, et « Head Fit » stocke les coefficients. Le cou : ses variations dans les scans
viennent surtout de leur pose, l'identité s'y estompe vers sa base (25 %, `NECK_IDENTITY_BASE`) et l'ajustement ne
l'utilise pas (la tête est rigide d'une vue à l'autre, le cou se plie différemment dans chacune). Sur des têtes synthétiques du modèle :
erreur hors visage −30 à 50 % par la prédiction (contre le simple suivi du visage), encore −35 % par l'ajustement.
Sans « Head From Face » : l'ancien comportement (la tête suit le visage, ajustement libre avec « Fit Radius »).

« Fit Shape » (bouton du nœud et du viewer, `face.fitShape`) : la forme du visage de l'acteur, comme la tête de
KeenTools FaceBuilder, trouvée sur les frames trackées de la plage (40 au plus, réparties, toutes les keyframes
comprises et comptées 3 fois) : une seule forme de repos pour le plan, ajustée avec la pose et l'expression (le rig,
raide) de chaque frame ; les frames vues sous plusieurs angles donnent la profondeur. Elle reste lisse (laplacien),
symétrique (sommets miroirs) et proche du maillage de départ ; alternance : les frames ajustées sur la forme
courante, puis la forme corrigée de leurs erreurs (linéarisé, un système creux). Gardée dans « Face Shape » (décalages
des sommets) ; le rig et les pins travaillent ensuite dessus (les pins restent posés sur le maillage de départ :
« Fit Shape » peut être relancé après de nouvelles keyframes) ; sortie « Shape » : la forme de repos, à écrire et
rebrancher dans « Face » d'autres FaceTrackers. « Clear Shape » la retire. Avec un visage branché en « Face » (un scan
de l'acteur de topologie MediaPipe), c'est lui la forme de repos : les points de MediaPipe ne donnent que la pose et
l'expression (le rig) ; en modèle « surface », chaque sommet est en revanche tiré vers son point.
« Temporal Smoothing » (frames, 1.5 par défaut) : le visage ajusté à chaque frame seule (`rawSolve`, en cache, la
pose partant de celle d'une frame voisine déjà résolue) est lissé dans le temps : la pose de la tête (rotations
moyennées en quaternions, translations) et la forme dans l'espace du visage (l'expression) sont moyennées sur les
frames voisines (gaussienne) ; plus de tremblement de la pose ni de partie du visage qui bascule en profondeur
d'une frame à l'autre. Les keyframes (frames avec pins) gardent leur propre ajustement. « Landmark Smoothing » lisse
les points 2D avant l'ajustement.
Frame de référence (« Use Reference Frame », « Reference Frame », bouton « Set Reference Frame » / étoile du
viewer) : la frame où le visage se voit le mieux (calée avec des pins) est ajustée sur le visage neutre, et son visage
ajusté devient la forme de l'acteur sur laquelle les autres frames sont ajustées (elles n'en changent que
l'expression) ; elles sont résolues de la référence vers l'extérieur, chacune depuis la pose de la précédente (pose
continue, résultat qui ne dépend pas de l'ordre d'affichage). La référence garde son ajustement (pas de lissage).
À chaque frame, un maillage de visage est calé dessus : la pose de la tête avec les points qui bougent peu
(front, arête du nez, coins des yeux, tempes ; robuste), puis, avec « Non Rigid », chaque sommet est amené à sa
profondeur sur le rayon de son point (les lèvres et les paupières suivent). « Smoothing » lisse les points dans le
temps. Entrée « Face » : n'importe quel maillage de topologie MediaPipe (468 ou 478 sommets dans l'ordre des points),
par exemple un scan de l'acteur retopologisé dessus, sa forme étant le visage neutre ; sans elle, le visage
canonique de MediaPipe. Sorties : « Face » (le maillage à chaque frame, dans le monde de la caméra), « Head Matrix »
(pose rigide de la tête : à brancher dans l'« Input Matrix » d'un ReadGeo), « Camera » (la caméra de la plaque :
entrée « Camera », sinon 35 mm à l'origine), l'image (maillage ou points dessinés). Ses pistes (« Tracks » :
features, all ou none ; nommées par région, lips_013, leftEye_033...) pilotent Roto, Stabilize et CornerPin comme
celles d'un Tracker. Installation : `pip install mediapipe` (matplotlib et OpenCV, demandés par les utilitaires
de dessin de MediaPipe, ne sont pas nécessaires : remplacés à l'import quand ils manquent, `face.importMediaPipe`) et
`python -m meshroom.live.neural.download facelandmarker` (modèle et visage canonique dans MESHROOM_MODELS_PATH).
Limite : les sommets gardent leur profondeur neutre (le mouvement des lèvres vers l'avant n'est pas suivi).

**Tête** (« Mesh » : head par défaut, ou face ; `face.canonicalHead`, `headOf`, `HeadFollow`, `facePart`) : le visage
canonique de MediaPipe (468 sommets, toujours en premier, dans l'ordre des points) avec oreilles, crâne et cou
(4 523 sommets de plus), tirés de la tête neutre d'ICT-FaceKit (licence MIT) : `meshroom/live/data/faceHead.npz`,
construit par `python -m meshroom.live.data.buildFaceHead canonical_face_model.obj generic_neutral_mesh.obj`
(alignement sur 68 points, découpe sous l'ovale du visage et sous le cou, déformation harmonique jusqu'à l'ovale ; un
sommet du bord par sommet de l'ovale est fusionné avec lui, les autres sont sur une arête de l'ovale et insérés dans
le polygone du visage le long de cette arête : maillage fermé hors du cou, sans fissure). Le fichier garde aussi le
visage canonique (licence Apache 2.0, `data/LICENSE-MediaPipe.md`) : il sert de visage par défaut quand MediaPipe
ne le fournit pas. Seul le visage est ajusté ; le reste de la tête le suit (`HeadFollow`) : une transformation
affine ajustée sur le haut du visage (au-dessus du bout du nez : pas la mâchoire) déplace toute la tête, les sommets
du bord restent sur leur arête de l'ovale, et ce que le visage fait de plus (mâchoire plus large, bouche ouverte)
s'estompe harmoniquement sur 5 cm. Une tête (sortie du FaceBuilder) peut être branchée en « Face » : son visage est
ajusté, ses oreilles et son crâne suivent ; les pins ne se posent que sur le visage. D'une frame à l'autre (une
expression), la tête suit le haut du visage rigidement (rotation et translation : le crâne garde sa forme) ; la
transformation affine ne sert qu'au changement de forme (la tête de l'acteur).

Pins en modèle `rig` : ensemble, ils pèsent autant que les points de MediaPipe (pose initiale et ajustement du rig) :
quelques pins suffisent à orienter la tête là où MediaPipe se trompe (profil, occultation). « Exact Pins » (activé
par défaut) : après le rig, un warp local rend chaque pin exact, mais seulement sur ce que le rig déplace
(`FaceRig.mobility` : bouche, mâchoire, paupières, sourcils, joues, bout du nez) ; un pin sur une partie rigide
(front, arête du nez, tempes, côtés de la tête) oriente la tête sans rien déformer. Désactivé : les pins orientent la
tête et bougent les contrôles du rig, rien d'autre.
Le rig ne replie jamais le maillage : chaque contrôle a sa limite (`faceRig.CONTROL_LIMITS`, en largeurs de visage :
nez et narines 2,5 %, joues et paupières 5 %, sourcils 8 %, lèvres 12 %, coins de la bouche 15 % ; la mâchoire ne se
ferme pas au-delà du repos) et son coût (nez ×4, joues ×2 : leurs points sont souvent faux de profil), les limites
sont dures (plus fortes que les pins), et si des contrôles tirés l'un contre l'autre retournent des triangles,
l'expression est atténuée jusqu'à ce qu'aucun ne le soit (`unfoldedFraction`). Le warp des pins exacts s'arrête de
même avant de replier le visage : un pin trop loin de ce que le visage peut faire n'est alors pas exact.
Pins rigides et pins d'expression (modèle `rig`) : un pin est rigide par défaut (cyan) : il ne fait que placer la
tête (rotation, position) ; le bouton « visage souriant » du viewer fait des pins sélectionnés des pins d'expression
(magenta, champ `"expression": true`) : ils déforment le visage (contrôles du rig, puis warp local avec « Exact
Pins »). Sans pin, la pose et le rig sont ajustés ensemble. Avec des pins, MediaPipe dit l'expression et les pins
disent où est la tête (`solveFace`) : l'expression est celle du visage ajusté sur les seuls points ; la pose est
ensuite ajustée seule avec cette expression, à partir des pins rigides (ensemble autant que les points) et des points
des parties rigides (points pondérés ×0,3 : la frame a des pins parce que MediaPipe s'y trompe) ; les pins
d'expression bougent enfin le rig à partir de cette expression, la pose gardée. Un pin rigide déplacé fait donc
tourner et bouger la tête sans rien déformer. Nez et narines : 1,5 % de la largeur du visage, coût ×8.

Suivi de la pose par la surface (« Follow Head Surface », activé par défaut, `face.HeadPoseTracker`) : avec Track
With flow ou mediapipe+flow, la pose de la tête n'est plus déduite des 478 points suivis un à un, mais suivie comme
le Resection suit une géométrie (`geotrack.stepPose` puis `alignPose` contre une frame d'ancrage) : jusqu'à 400 points
texturés (Shi-Tomasi) des parties rigides du visage (front, arête du nez, tempes, pommettes : rigidité ≥ 0,6), suivis
par le flow (Lucas-Kanade ou RAFT), pose robuste, parties masquées exclues. La pose est gardée avec chaque frame
(champ `"p"` de « faceTrack ») et `solveFace(pose=...)` la garde : les points ne donnent plus que l'expression (rig
seul). Une keyframe avec des pins rigides est placée par ses pins. Surface perdue (trop peu de points texturés) : la
frame suit les points comme avant.
Un point suivi sur dix est gardé avec la frame (champ `"t"` : x, y en pixels, 1 s'il colle à la pose, 0 s'il est
rejeté) et affiché (« Show Surface Tracks ») : vert ou rouge, pour voir sur quoi la pose s'accroche.

Affichage du maillage en quads (wireframe du FaceTracker et du FaceBuilder, `geoRender.quadEdges`) : les triangles
sont appariés en quads convexes, presque plans, aux angles proches de 90° (meilleures paires d'abord) et la diagonale
n'est pas tracée ; affichage seul, la géométrie reste en triangles.

Détection : un visage couché (roll de plus de 45°) est cherché dans le crop tourné d'un quart ou d'un demi-tour
(`_detectUpright` : d'abord selon le roll du visage de la frame précédente, `faceRoll` ; « Detect Lying Faces »
pour le couper), les points revenant dans l'image ; un crop sombre est éclairci d'après les hautes lumières de son
milieu (×8 au plus, `_exposed`) : un ciel clair au bord ne l'empêche plus.

**FaceBuilder** (Comp 3D, `FaceNodes.py`, `face.buildShape`) : la tête d'un acteur à partir de photos, comme KeenTools
FaceBuilder. Entrées « View 1 » à « View 4 » : chacune une photo ou une séquence (chaque frame une vue, « Max
Frames » au plus, réparties) ; tailles et focales différentes acceptées. « Build Head » : MediaPipe trouve les points
du visage dans chaque vue (`detectViews`, gardés dans « Views »), puis la forme est ajustée sur toutes les vues à la
fois (`face.fitShape`, chaque vue avec sa pose, son expression par le rig et sa taille d'image), la focale de chaque
vue estimée (`estimateFocal` : la focale où le visage colle le mieux à ses points, légère préférence pour 50 mm ; puis
réestimée sur la forme ajustée) ou donnée (« Focal Length », mm sur 36 mm). « Fit Again » réajuste sur les points
gardés (après un changement de « Smoothness », « Symmetry », « Stay Near Base »). Sorties : « Head » (la tête
déformée, espace du visage, en cm), « Original » (la tête de base), « Camera » (la caméra de la vue affichée, la tête
à l'origine) et l'image de la vue choisie par « Show View » avec la tête dessinée. Entrée « Base » : un autre visage
ou une tête de topologie MediaPipe. Brancher « Head » dans « Face » d'un FaceTracker pour suivre la tête de l'acteur.

**Plage de tracking** (« Limit Track Range », « Track First », « Track Last » : Tracker, CameraTracker, Resection,
FaceTracker, Roto ; `params.trackRangeParams` / `trackRange`) : les actions de tracking (boutons Track du viewer,
Autotrack, Track Geometry et Refine, Track Face, Track Shapes) ne calculent que dans cette plage ; une frame courante
hors de la plage est refusée.

### 4.14 GPU et réseaux de neurones

**GPU** (`meshroom/live/gpu.py`, `gpuOps.py`) : les nœuds Blur, Sharpen, Erode, Median, ZDefocus, OpticalFlow,
Retime, TimeWarp, Tracker, CameraTracker et Resection ont « Use GPU » (coché par défaut). Avec PyTorch et un GPU
CUDA (ou MPS sur macOS), le calcul y est fait ; sinon, ou en cas d'échec (mémoire, opération non supportée), sur le
CPU, avec le même résultat (avertissement une fois par opération dans le log). Ce qui passe sur le GPU : flous
(gaussien et trois boîtes), érosion / dilatation, médiane (par bandes), convolutions FFT du ZDefocus, flot optique
Lucas-Kanade pyramidal, et la corrélation des motifs de toutes les pistes à la fois (Tracker, pins, autotrack,
suivi de géométrie par patches). Le Roto (« Track Shapes With Flow ») utilise le GPU s'il y en a un.
`MESHROOM_GPU=0` l'interdit partout, `MESHROOM_GPU_DEVICE=1` choisit la carte.

**Réseaux** (`meshroom/live/neural/`, optionnels, PyTorch) :

| Réseau | Licence | Où | Option |
|--------|---------|----|--------|
| RAFT (torchvision `raft_large`) | BSD-3 | OpticalFlow, suivi de géométrie | « Method » : raft ; « Track Geometry With » : raft |
| BootsTAPIR (tapnet) | Apache-2.0 / CC-BY-4.0 | Tracker, pins de la Resection | « Point Tracker » : tapir |
| CoTracker3 | **CC-BY-NC-4.0, non commercial** | Tracker, pins | « Point Tracker » : cotracker |
| LightGlue + ALIKED | Apache-2.0 / BSD-3 | CameraTracker, Align de la Resection | « Features » : lightglue ; « Align With » : lightglue |
| MegaPose (megapose6d ou happypose) | Apache-2.0 | Align de la Resection | « Align With » : megapose (GPU CUDA) |

FoundationPose n'est pas intégré : il lui faut une image de profondeur (RGB-D) en plus de l'image, ce que n'ont pas
les plaques, et sa licence est non commerciale. MegaPose a un modèle RGB seul.

- **Point Tracker** (tapir / cotracker) : le réseau voit 32 (48) frames à la fois, suit les points dans les
  mouvements rapides et le flou, et dit quand un point est caché (sa piste s'arrête là) ; chaque position est
  affinée sur l'image (motif de la frame précédente cherché tout près, sous-pixel) ; la fenêtre suivante repart des
  positions trouvées (`tracking.trackManyNeural`).
- **Features : lightglue** : les features ALIKED de chaque frame appariées à celles de la suivante par LightGlue ;
  une feature réappariée prolonge sa piste (`camera.autoTrackMatched`).
- **Align With** : patches (comme avant : la géométrie doit être placée à peu près), silhouette (sans keyframe ni
  réseau, `geotrack.alignToSilhouette` : le contour de la géométrie vu depuis la pose actuelle est recalé sur les
  bords de l'image ; chaque point du contour cherche le bord le plus fort le long de sa normale, à la RAPiD, si bien
  que la texture à l'intérieur de l'objet ne l'attire pas ; la pose est résolue sur la distance le long des normales
  (Levenberg-Marquardt, poids de Tukey), du flou au net, sur « Silhouette Search » pixels au plus (40 par défaut,
  avancé) ; il faut un objet qui se détache de son fond et une géométrie placée à moins de cette distance ; un objet
  symétrique peut se recaler sur une pose symétrique ; l'entrée « Mask » de la Resection limite aussi le contour
  recalé à sa partie blanche), lightglue (les features de la
  keyframe appariées dans la frame courante puis posées sur la géométrie : marche loin de la bonne position et sous
  un autre angle, `geotrack.alignWithMatches`), megapose (la pose du maillage trouvée dans l'image depuis la boîte où
  la pose actuelle le montre, sans keyframe). Dans tous les cas, s'il y a une keyframe, la pose est ensuite affinée sur les patches de la
  keyframe la plus proche.
- Un réseau absent (PyTorch, paquet, poids) : le nœud prend la méthode classique (avertissement dans le log ; pour
  Align, un message d'erreur dit ce qui manque).

**Installation** : PyTorch (`rez env pytorch-2.5.1 ...`, avec torchvision pour RAFT) ; les paquets
`tapnet` (git+https://github.com/google-deepmind/tapnet), `lightglue` (git+https://github.com/cvg/LightGlue),
`cotracker` (git+https://github.com/facebookresearch/co-tracker), `megapose6d` ou `happypose`. Les poids sont lus
dans le dossier `MESHROOM_MODELS_PATH` (jamais téléchargés pendant que Meshroom tourne) ; le remplir une fois, depuis
une machine qui a internet :

```
python -m meshroom.live.neural.download --path /chemin/partage/meshroom-models            # tous sauf CoTracker
python -m meshroom.live.neural.download --path /chemin/partage/meshroom-models cotracker  # non commercial
export MESHROOM_MODELS_PATH=/chemin/partage/meshroom-models
```

Les noyaux GPU sont testés contre leurs versions CPU sur le device « cpu » de PyTorch (`tests/test_gpu.py`, sautés
sans PyTorch).

### Couches différées

Une couche d'image peut être différée (`imageBuffer.LazyPixels` : lue ou calculée à sa première utilisation, une
seule fois, thread-safe ; sa forme est connue sans la calculer). Le Read (OpenImageIO) lit les en-têtes de toutes les
parts d'un EXR multipart puis seulement la couche principale ; chaque autre couche est lue quand un nœud s'en sert,
seule (sa part, ses canaux : `_readChannels`). `layer()`, `layers()`, `rgba()` donnent les pixels (calculés au
besoin), `rawLayers()` les couches telles quelles pour les transmettre sans les calculer ; `mapLayersDeferred(func)`
applique une fonction pure à la couche principale tout de suite, aux autres plus tard (crop de la région d'intérêt,
proxy, recadrage des couches de B par le Merge). Le cache compte une couche différée comme calculée.

### Mesure des performances

`python -m meshroom.live.benchmark projet.mg` (`meshroom/live/benchmark.py`) : rend les branches des Write du projet
(dans les subgraphs aussi : `compositing_1`), ou les nœuds donnés (`--node compositing_1/Merge_1`, répétable), sur
`--frames` (`1001-1010`, `1001,1005`, `1001-1100x10` ; par défaut dix frames depuis le début de la timeline), une fois
avec un cache vide (« cold » : tout calculé) puis à nouveau (« warm » : ce que donne le cache). Tableau par nœud, du
plus cher au moins cher : rendus, rendus pris dans le cache, temps propre (ses entrées exclues, aussi pour les nœuds
temporels qui rendent leurs entrées eux-mêmes), temps par rendu, part, taille des images produites, octets envoyés au
GPU et ramenés (`gpuOps`, `geoRender`, RAFT : `profiling.uploaded / downloaded`) ; `--memory` : mémoire allouée par
chaque nœud au-delà de ce qui était déjà tenu (tracemalloc, plus lent) et pic de VRAM (PyTorch) ; `--python-profile N` :
les N fonctions Python les plus chères (cProfile) ; `--no-gpu`, `--proxy`, `--cache-gb`, `--json` (tout : temps par
frame, nœuds, pics). Le profileur (`profiling.Profiler`) se pose sur n'importe quel `Evaluator` (`evaluator.profiler`).

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
