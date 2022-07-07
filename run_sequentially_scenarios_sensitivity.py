import os
import time


def main(*args, cpus=10, runs=4):
    with open('errors_on_sensitivity.txt', 'a') as handler:
        for p in args:
            try:
                print(p)
                os.system(p)
            except Exception as e:
                handler.write(f'{time.asctime()}: {e} \n')


if __name__ == '__main__':
    r = 100
    c = 14
    c1 = f"python main.py {c} {r} scenarios all"

    c2 = f"python runner.py {c} {r}"

    main(*[c1, c2], cpus=c, runs=r)
