for i in $(seq 1 1000)
do
    ../../mosaic.py sum.py | grep SUM | grep stdout >> result.txt
done