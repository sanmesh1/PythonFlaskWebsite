import os, time, sys

path = "static"

now = time.time()
numMinutes = 0

for f in os.listdir(path):
    f1 = os.path.join(path, f)
##    print("Is this a dir? ", os.path.isdir(f1))
##    print("f1 path: ", f1)
##    print("time in seconds", os.stat(f1).st_mtime)
##    print("current time", now)
##    print((now - os.stat(f1).st_mtime) )
    if (now - os.stat(f1).st_mtime) > 60*numMinutes:
        if os.path.isfile(f1) and os.path.isdir(f1) == False:
            os.remove(f1)
