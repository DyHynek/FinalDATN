import multiprocessing
import subprocess
import os

def run_script1():
    subprocess.run(["python", os.path.join(os.path.dirname(__file__), "predict.py")])

def run_script2():
    subprocess.run(["python", os.path.join(os.path.dirname(__file__), "app.py")])

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_script1)
    p2 = multiprocessing.Process(target=run_script2)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
