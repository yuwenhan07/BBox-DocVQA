python pdf2png_parallel.py ../../data/arxiv_pdf_725/   -o ../../data/arxiv_png_725/   -w 128 --dpi 300

python get_arxiv_id_speed.py --majors --per-field 100 --start-date 2023-01-01 --end-date 2024-12-31 --paper-flags "-p -n 4" --workers 20 --timeout 900 --skip-existing --base-dir ../../data/downloads
