# ragulate_bio/cli.py
def main():
    from . import about
    info = about()
    print(f"{info['name']} {info['version']}: {info['summary']}")
