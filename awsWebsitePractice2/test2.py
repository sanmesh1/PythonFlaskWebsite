import os, time, sys

now = time.time()
path = "static"
numMinutes = 20

for f in os.listdir(path):
    f1 = os.path.join(path, f)
    if (now - os.stat(f1).st_mtime) > 60*numMinutes:
        if os.path.isfile(f1) and os.path.isdir(f1) == False and f1.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(f1)

now = time.time()
path = "static/static"            
for f in os.listdir(path):
    f1 = os.path.join(path, f)
    if (now - os.stat(f1).st_mtime) > 60*numMinutes:
        if os.path.isfile(f1) and os.path.isdir(f1) == False and f1.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(f1)

now = time.time()
path = "static/static/upload"          
for f in os.listdir(path):
    f1 = os.path.join(path, f)
    if (now - os.stat(f1).st_mtime) > 60*numMinutes:
        if os.path.isfile(f1) and os.path.isdir(f1) == False and f1.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(f1)
