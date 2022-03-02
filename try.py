def try1(b):
    b.append('lol')

def loop():
    b = []
    for i in range(10):
        try1(b)
    print(b)

loop()