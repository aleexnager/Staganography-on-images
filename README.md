PROYECTO DE ESTEGANOGRAFÍA

Para ejecurtar:
python3 main.py --mode encode --input data/input/yak.png --output data/output/yak_lsb.png --message data/messages/msg1.txt

Y para decodificar:
python3 main.py --mode decode --input data/output/yak_lsb.png

Para ejecutar los tests:
pytest tests/test_lsb.pyt
pytest tests/test_lsb_optimized.py
pytest tests/test_lsb.py tests/test_lsb_optimized.py

Para generar tablas de diferencias
python generate_differences.py
