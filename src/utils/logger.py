def log(text=None) -> None:
    if text is None:
        print(f"[INFO]", flush=True)
    else:
        print(f"[INFO] {text}", flush=True)


def error(text=None) -> None:
    if text is None:
        print(f"[ERROR]", flush=True)
    else:
        print(f"[ERROR] {text}", flush=True)