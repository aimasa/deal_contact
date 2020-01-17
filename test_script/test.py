import os
print("1:",os.path.join('aaaa','sd/bbb','ccccc.txt'))
print("2:",os.path.join('bbb','aaaa','./sd/bbb','ccccc.txt'))
print("3:",os.path.join('bbb','/aaaa','./sd/bbb','ccccc.txt'))
print("4:",os.path.join('aaaa','E:/sd/bbb','ccccc.txt'))
print("5:",os.path.join('aaaa','./sd/bbb','ccccc.txt'))
print("6:",os.path.join('aaaa','/aaoo','/aaoo/xxx','/sd/bbb','ccccc.txt'))
