#!/bin/bash
echo "Fetching Ascession $1"
fastq-dump  --split-files --gzip -O fq  $1
echo "Aligning paired reads"
bwa aln -t 7 hg19bwaidx fq/$1_1.fastq.gz > sai/$1_1.sai
bwa aln -t 7 hg19bwaidx fq/$1_2.fastq.gz > sai/$1_2.sai
echo "Producing $1.sam"
bwa sampe hg19bwaidx sai/$1_1.sai sai/$1_2.sai fq/$1_1.fastq.gz fq/$1_2.fastq.gz > sam/$1.sam
echo "Producing indexed $1.bam"
samtools view -bT hg19.fa sam/$1.sam | samtools sort -O bam -o bam/$1.bam -T temp
samtools index bam/$1.bam

echo "Pipeup"
bcftools mpileup -f hg19.fa -O b --threads 7 bam/$1.bam > bcf/$1.bcf
bcftools index bcf/$1.bcf
echo "Pick chrM"
bcftools call -r chrM -c -O b bcf/$1.bcf > chrM/bcf/$1.chrM.bcf
bcftools index  chrM/bcf/$1.chrM.bcf
echo "Generate fasta chrM/fa/$1.chrM.fa"
samtools faidx hg19.fa chrM:1-16571 | bcftools consensus chrM/bcf/$1.chrM.bcf > chrM/fa/$1.chrM.fa





