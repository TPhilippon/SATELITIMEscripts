#!/bin/bash
# Lancer le script sur le serveur /bin/bash /media/pressions/SATELITIME/scripts/video/5_morph_bash_loop.sh 

echo "Bash version ${BASH_VERSION}..."
for i in {0..578}
do
	let j=$i+1
	printf -v f "%04d" $i
	printf -v g "%04d" $j
	echo $f
	echo $g
	start=`date +%s.%N`
	convert multi$f.tif multi$g.tif -morph 5 morph/morph$f.png
	end=`date +%s.%N`
	runtime=$( echo "$end-$start"| bc -l)
	echo $runtime
done
