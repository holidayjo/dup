DUP project.
This is a private project for dup.

I clearly mention that I copied yolov7 official repository in the link below.

https://github.com/WongKinYiu/yolov7.

For annotation, I used 2 base repositoreis below (LabelImg and Yolo-annotation-Tool-New).

https://github.com/heartexlabs/labelImg

https://github.com/ManivannanMurugavel/Yolo-Annotation-Tool-New-

Based on the projects above, I modify the codes as my preference.

All dataset are not included in the repository even though it is private repository for 2 reasons - size and privacy.

The results from the mmdetection is converted to yolo format which is txt file per image throughout result_to_yolo.py file.


* commands (Run at yolov9 folder.)
python train_dual.py --workers 8 --device 0 --batch 50 --data data/udp.yaml --img 320 --cfg models/detect/yolov9-c.yaml --weights 'yolov9-c' --name yolov9-c --hyp data/hyps/udp.yaml --min-items 0 --epochs 180 --close-mosaic 15

python train.py --workers 8 --device 0 --batch 80 --data /mnt/Documents/Dad/github/DUP/yolov9_udp/data/udp.yaml --img 320 --cfg models/detect/udp_gelan-c.yaml --weights 'gelan-c.pt' --name gelan-c --hyp hyps/udp.yaml --epochs 200