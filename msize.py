import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python msize.py <m> <k> <n>")
        sys.exit(1)

    try:
        m = int(sys.argv[1])
        k = int(sys.argv[2])
        n = int(sys.argv[3])
    except ValueError:
        print("All arguments must be integers.")
        sys.exit(1)

    # tile_m = 64
    # tile_k = 256
    # tile_n = 64

    size_A = m * k
    size_B = k * n
    size_C = m * n * 4  # C elem is int32_t = 4B

    print(f"Received arguments: m={m}, k={k}, n={n}")
    print(f"Size of A: {size_A}B = {size_A / (1024)}KB")
    print(f"Size of B: {size_B}B = {size_B / (1024)}KB")
    print(f"Size of C: {size_C}B = {size_C / (1024)}KB")
    print(f"Total size: {size_A + size_B + size_C}B = {(size_A + size_B + size_C) / (1024 * 1024)}MB")

if __name__ == "__main__":
    main()