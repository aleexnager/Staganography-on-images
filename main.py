# main.py

import argparse
from stegano.lsb import encode_lsb, decode_lsb
from stegano.lsb_optimized import encode_lsb_optimized, decode_lsb_optimized
from stegano.lsb_adaptive import encode_lsb_adaptive, decode_lsb_adaptive
from stegano.utils import load_text_file

def main():
    parser = argparse.ArgumentParser(description="LSB Steganography CLI")
    parser.add_argument("--mode", choices=["encode", "decode"], required=True, help="Modo: encode o decode")
    parser.add_argument("--input", required=True, help="Ruta de la imagen de entrada")
    parser.add_argument("--output", help="Ruta de la imagen de salida (solo para encode)")
    parser.add_argument("--message", help="Ruta del archivo de texto con el mensaje (solo para encode)")
    parser.add_argument("--delimiter", default="#####", help="Delimitador de fin de mensaje")
    parser.add_argument(
        "--method",
        choices=["lsb", "lsb_opt", "lsb_adapt"],
        default="lsb",
        help="Método: lsb, lsb_opt o lsb_adapt"
    )
    parser.add_argument("--original", help="Ruta de la imagen original (solo para lsb_adapt decode)")
    # Parámetros adaptativos
    parser.add_argument(
        "--thresholds",
        nargs=2,
        type=float,
        default=(0.5, 0.75),
        help="Umbrales de entropía t0 t1 para lsb_adapt"
    )
    parser.add_argument(
        "--bits-per-channel",
        nargs=3,
        type=int,
        default=(0, 1, 2),
        help="Bits por canal (b0 b1 b2) para lsb_adapt"
    )

    args = parser.parse_args()

    if args.mode == "encode":
        if not args.output or not args.message:
            print("Para 'encode' necesitas --output y --message")
            return
        message = load_text_file(args.message)
        if args.method == "lsb":
            encode_lsb(args.input, message, args.output, delimiter=args.delimiter)
        elif args.method == "lsb_opt":
            encode_lsb_optimized(args.input, message, args.output, delimiter=args.delimiter)
        else:  # lsb_adapt
            encode_lsb_adaptive(
                args.input,
                message,
                args.output,
                window_size=3,
                thresholds=tuple(args.thresholds),
                bits_per_channel=tuple(args.bits_per_channel),
                delimiter=args.delimiter
            )
        print(f"Mensaje ocultado en {args.output}")

    else:  # decode
        if args.method == "lsb":
            message = decode_lsb(args.input, delimiter=args.delimiter)
        elif args.method == "lsb_opt":
            message = decode_lsb_optimized(args.input, delimiter=args.delimiter)
        else:  # lsb_adapt
            if not args.original:
                print("Para 'lsb_adapt' decode necesitas --original")
                return
            message = decode_lsb_adaptive(
                args.input,
                args.original,
                window_size=3,
                thresholds=tuple(args.thresholds),
                bits_per_channel=tuple(args.bits_per_channel),
                delimiter=args.delimiter
            )
        print("Mensaje extraído:")
        print(message)

if __name__ == "__main__":
    main()
