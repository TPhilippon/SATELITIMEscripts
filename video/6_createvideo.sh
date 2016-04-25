avconv -r 30  -i A%5d.png -vf "transpose=2,transpose=2" -c:v libx264 -crf 20 -pix_fmt yuv420p chlor.mp4
