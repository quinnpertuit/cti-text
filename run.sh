cd /mnt/clustering/data
tar -xzf "Resumes Raw Text - 1.02m Resumes, 2020-09-02.csv.gz"
rm *gz
mv *csv 1.02m_resumes_dataset.csv
cd /mnt/clustering
python model/main.py --fpath data/1.02m_resumes_dataset.csv --ntopic 5000 --method LDA_BERT --samp_size 200000 --savepng 0