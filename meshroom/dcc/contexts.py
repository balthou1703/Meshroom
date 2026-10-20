"""
Contexts of Houdini, Maya and Nuke for the Meshroom subgraphs (see docs/subgraph.md of Meshroom): registered when
the nodes of this plugin are loaded.
"""
import importlib
import logging
import os

from meshroom.core.subgraph import registerContext
from meshroom.core.subgraph.contexts import SceneContext


def _importable(moduleName: str) -> bool:
    try:
        importlib.import_module(moduleName)
        return True
    except ImportError:
        return False


class HoudiniContext(SceneContext):
    """
    Houdini: `ctx.context.session` is the hou module. Available in hython and in a Houdini session; from
    Meshroom or a farm, the subgraph runs in hython (the interpreter of the Houdini nodes of the subgraph,
    else MESHROOM_HYTHON). The current frame follows the frame loops. Scene: hou.hipFile.load / save are
    intercepted during a run (see SceneContext).
    """
    interpreterVariable = "MESHROOM_HYTHON"
    defaultInterpreter = "hython"
    executables = ("hython",)

    @classmethod
    def isAvailable(cls) -> bool:
        return _importable("hou")

    def open(self):
        import hou
        self.session = hou
        super().open()

    def currentScene(self) -> str:
        return self.session.hipFile.path()

    def _installGuard(self):
        context = self

        def loadWrapper(original):
            def load(file_name, *args, **kwargs):
                if context.skipLoad(file_name):
                    return None
                return original(file_name, *args, **kwargs)
            return load

        def saveWrapper(original):
            def save(file_name=None, *args, **kwargs):
                context.deferSave(file_name)
            return save

        self._patch(self.session.hipFile, "load", loadWrapper)
        self._patch(self.session.hipFile, "save", saveWrapper)

    def _load(self, path):
        # In a user session, Houdini asks whether to save the changes of the open scene
        load = self._original("load", self.session.hipFile.load)
        load(path, suppress_save_prompt=self.ownsSession, ignore_load_warnings=True)

    def _save(self, path):
        self._original("save", self.session.hipFile.save)(path)

    def _clear(self):
        self.session.hipFile.clear(suppress_save_prompt=True)

    def beginStep(self, iteration, frame):
        if frame is not None:
            self.session.setFrame(frame)

    @staticmethod
    def preloadHostUsd():
        """
        In a Houdini process, import the USD Python modules (pxr) of Houdini before the node types are loaded,
        so that a plugin importing pxr gets the one of Houdini and not another USD found in PYTHONPATH (two USD
        libraries in one process make Houdini abort). Does nothing outside of Houdini or if pxr is imported.
        """
        import sys
        hfs = os.environ.get("HFS")
        if "hou" not in sys.modules or "pxr" in sys.modules or not hfs:
            return
        pythonLibs = os.path.join(hfs, "houdini", f"python{sys.version_info[0]}.{sys.version_info[1]}libs")
        if not os.path.isdir(os.path.join(pythonLibs, "pxr")):
            return
        if pythonLibs in sys.path:
            sys.path.remove(pythonLibs)
        sys.path.insert(0, pythonLibs)
        try:
            importlib.import_module("pxr.Tf")
        except ImportError as e:
            logging.warning(f"[SubGraph] Cannot preload the USD modules of Houdini: {e}")


def _samePath(a, b) -> bool:
    return bool(a) and bool(b) and os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


class MayaContext(SceneContext):
    """
    Maya: `ctx.context.session` is the maya.cmds module. Available in mayapy (initialized as standalone)
    and in a Maya session (the interpreter of the Maya nodes of the subgraph, else MESHROOM_MAYAPY).
    The current time follows the frame loops. Scene: cmds.file(open=...) / cmds.file(save=...) are
    intercepted during a run (see SceneContext).
    """
    interpreterVariable = "MESHROOM_MAYAPY"
    defaultInterpreter = "mayapy"
    executables = ("mayapy",)

    @classmethod
    def isAvailable(cls) -> bool:
        return _importable("maya.cmds")

    def open(self):
        import maya.cmds as cmds
        self._initialized = False
        if not hasattr(cmds, "about"):
            # mayapy: initialize Maya
            import maya.standalone
            maya.standalone.initialize(name="python")
            self._initialized = True
        self.session = cmds
        super().open()

    def currentScene(self) -> str:
        return self._original("file", self.session.file)(query=True, sceneName=True) or ""

    def _installGuard(self):
        context = self

        def fileWrapper(original):
            def file(*args, **kwargs):
                query = kwargs.get("query") or kwargs.get("q")
                if not query and (kwargs.get("open") or kwargs.get("o")) and args:
                    if context.skipLoad(args[0]):
                        return args[0]
                elif not query and (kwargs.get("save") or kwargs.get("s")):
                    context.deferSave()
                    return context.currentScene()
                return original(*args, **kwargs)
            return file

        self._patch(self.session, "file", fileWrapper)

        # Maya is initialized (a Maya session, or mayapy initialized by `open`): the nodes written for mayapy
        # initialize / uninitialize it themselves, which fails in a Maya session ("can not be called inside
        # of Maya") or would end the session for the next nodes
        try:
            import maya.standalone as standalone
        except ImportError:
            return

        def initializeWrapper(original):
            def initialize(*args, **kwargs):
                logging.info("[SubGraph] maya.standalone.initialize: Maya is already initialized.")
            return initialize

        def uninitializeWrapper(original):
            def uninitialize(*args, **kwargs):
                logging.info("[SubGraph] maya.standalone.uninitialize: done at the end of the subgraph.")
            return uninitialize

        self._patch(standalone, "initialize", initializeWrapper)
        self._patch(standalone, "uninitialize", uninitializeWrapper)

    def _load(self, path):
        self._original("file", self.session.file)(path, open=True, force=True)

    def _save(self, path):
        file = self._original("file", self.session.file)
        if not _samePath(self.currentScene(), path):
            file(rename=path)
        file(save=True, force=True, type="mayaAscii" if str(path).lower().endswith(".ma") else "mayaBinary")

    def _clear(self):
        self._original("file", self.session.file)(new=True, force=True)

    def close(self):
        super().close()
        if self.ownsSession and self._initialized:
            import maya.standalone
            maya.standalone.uninitialize()

    def beginStep(self, iteration, frame):
        if frame is not None:
            self.session.currentTime(frame, edit=True)


class NukeContext(SceneContext):
    """
    Nuke: `ctx.context.session` is the nuke module. Available in a Nuke session and with "nuke -t" (the
    interpreter of the Nuke nodes of the subgraph, else MESHROOM_NUKE). The current frame follows the
    frame loops. Script: nuke.scriptOpen / scriptSave / scriptSaveAs are intercepted during a run (see
    SceneContext).
    """
    interpreterVariable = "MESHROOM_NUKE"
    defaultInterpreter = "nuke -t"
    executables = ("nuke",)

    @classmethod
    def isAvailable(cls) -> bool:
        return _importable("nuke")

    def open(self):
        import nuke
        self.session = nuke
        super().open()

    def currentScene(self) -> str:
        try:
            return self.session.scriptName()
        except RuntimeError:
            # Untitled script
            return ""

    def _installGuard(self):
        context = self

        def openWrapper(original):
            def scriptOpen(filename, *args, **kwargs):
                if context.skipLoad(filename):
                    return None
                return original(filename, *args, **kwargs)
            return scriptOpen

        def saveWrapper(original):
            def scriptSave(filename=None, *args, **kwargs):
                context.deferSave(filename or None)
                return True
            return scriptSave

        def saveAsWrapper(original):
            def scriptSaveAs(filename=None, *args, **kwargs):
                context.deferSave(filename or None)
            return scriptSaveAs

        self._patch(self.session, "scriptOpen", openWrapper)
        self._patch(self.session, "scriptSave", saveWrapper)
        self._patch(self.session, "scriptSaveAs", saveAsWrapper)

    def _load(self, path):
        if self.ownsSession:
            self.session.scriptClear()
        self._original("scriptOpen", self.session.scriptOpen)(path)

    def _save(self, path):
        self._original("scriptSaveAs", self.session.scriptSaveAs)(path, 1)

    def _clear(self):
        self.session.scriptClear()

    def beginStep(self, iteration, frame):
        if frame is not None:
            self.session.frame(frame)


registerContext("houdini", HoudiniContext)
registerContext("maya", MayaContext)
registerContext("nuke", NukeContext)
