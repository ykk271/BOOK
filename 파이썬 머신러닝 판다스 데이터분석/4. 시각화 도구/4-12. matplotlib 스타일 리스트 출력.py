import matplotlib

colors = {}

for name, hex in matplotlib.colors.cnames.items():
    colors[name] = hex

print(colors)
