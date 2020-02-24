CUDA_VISIBLE_DEVICES=0 gunicorn -t 360 -b 0.0.0.0:8111 ocr_server:app
tail -f /dev/null