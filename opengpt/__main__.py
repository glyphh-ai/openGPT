"""python -m opengpt            start the app
python -m opengpt --lan      start it for every device on your network
python -m opengpt fetch      download TinyStories into data/
python -m opengpt train k=v  train in the terminal, e.g. steps=2000 n_layer=2
python -m opengpt check      re-check the gradients of model.py
"""

import sys
import time


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args.pop(0) if args and not args[0].startswith("-") else "serve"

    if cmd == "serve":
        from . import server
        port = int(args[args.index("--port") + 1]) if "--port" in args else 8642
        # --lan lets other devices on your network open the app; it has no password
        host = args[args.index("--host") + 1] if "--host" in args else "0.0.0.0" if "--lan" in args else "127.0.0.1"
        server.serve(host=host, port=port)
    elif cmd == "fetch":
        from . import data
        print("downloading TinyStories (about 19 MB) ...")
        print("saved", data.fetch_tinystories())
    elif cmd == "check":
        from . import api
        for n_layer, n_head in [(1, 1), (2, 2), (3, 4)]:
            r = api.gradcheck({}, dict(n_layer=n_layer, n_head=n_head))
            print(f"blocks {n_layer}  heads {n_head}  largest gap {r['error']:.1e}  "
                  f"{'ok' if r['ok'] else 'WRONG'}")
    elif cmd == "train":
        from . import trainer
        run = trainer.Run(dict(a.split("=", 1) for a in args)).start()
        seen = 0
        while run.alive or seen < len(run.evals):
            for e in run.evals[seen:]:
                print(f"step {e['step']:6d}  train loss {e['train_loss']:.3f} acc {e['train_acc']:.1%}"
                      f"  |  val loss {e['val_loss']:.3f} acc {e['val_acc']:.1%}")
            seen = len(run.evals)
            time.sleep(0.2)
        if run.error:
            sys.exit(run.error)
        print("\nsample:", run.samples[-1]["text"] if run.samples else "")
        print("saved in", run.dir)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
