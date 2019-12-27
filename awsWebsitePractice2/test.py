import os, time, sys

path = "static"

now = time.time()
numMinutes = 10

for f in os.listdir(path):
    f1 = os.path.join(path, f)
    print(f1)
    print(os.stat(f1).st_mtime)
    print(now)
    if os.stat(f1).st_mtime < now - 60*numMinutes:
        if os.path.isfile(f1):
            print("old file is: ", f1)
            print("hours since last modified = ", (now - os.stat(f1).st_mtime)/(60*60))
            #os.remove(os.path.join(path, f1))
