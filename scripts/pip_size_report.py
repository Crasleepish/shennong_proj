import os
import sys
import site

def get_site_packages():
    paths = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    paths += [site.getusersitepackages()]
    return [p for p in paths if os.path.isdir(p)]

def get_size(path):
    total = 0
    for root, dirs, files in os.walk(path, onerror=None):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except:
                pass
    return total

def main():
    print("Scanning pip packages...\n")
    dirs = get_site_packages()

    results = []
    for d in dirs:
        for pkg in os.listdir(d):
            full = os.path.join(d, pkg)
            size = get_size(full)
            results.append((pkg, size))

    results.sort(key=lambda x: x[1], reverse=True)

    for pkg, size in results:
        print(f"{pkg:40}  {size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
