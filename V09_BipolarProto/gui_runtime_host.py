import threading
from pathlib import Path


_SOURCE = Path(__file__).with_name("gui_app.py")
_CODE = compile(_SOURCE.read_bytes(), str(_SOURCE), "exec")
_LOCK = threading.Lock()
_OWNER_NAMESPACE = None


def _release_owner(namespace):
    global _OWNER_NAMESPACE
    if _OWNER_NAMESPACE is namespace:
        _OWNER_NAMESPACE = None


def serve_gui():
    global _OWNER_NAMESPACE

    namespace = {
        "__builtins__": __builtins__,
        "__file__": str(_SOURCE),
        "__name__": f"dmps_gui_session_{id(threading.current_thread())}",
        "__package__": None,
    }
    with _LOCK:
        try:
            exec(_CODE, namespace)
        except Exception:
            bridge = namespace.get("runtime_bridge")
            panel = namespace.get("pn")
            shutdown = namespace.get("safe_shutdown")
            if callable(shutdown):
                try:
                    shutdown("runtime initialization failed")
                except Exception:
                    pass
            if panel is not None and bridge is not None:
                key = namespace.get("RUNTIME_BRIDGE_KEY")
                if key and panel.state.cache.get(key) is bridge:
                    panel.state.cache.pop(key, None)
            raise
        if namespace.get("runtime_owner"):
            _OWNER_NAMESPACE = namespace
            namespace["runtime_bridge"]["release_owner"] = (
                lambda: _release_owner(namespace)
            )
