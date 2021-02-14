import archs.core.resnet

if __name__ == "__main__":
    for mode in ["raw", "dunder_file", "importlib"]:
        try:
            arch = archs.core.resnet.resnet56(pretrained=True, mode=mode)
            print("Success.")
        except FileNotFoundError:
            print(f"Error: loading weight failed under mode={mode}.")
