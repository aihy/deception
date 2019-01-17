python -u main.py \
	-a resnet34 \
	-j 4 \
	-b 500 \
	--resume model_best.pth.tar \
	-e

