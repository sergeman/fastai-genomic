#!/bin/bash

#samtools view bam/$1.bam chrM -b > chrM/$1.chrM.bam
#bcftools mpileup -f hg19.fa chrM/$1.chrM.bam -O b >  chrM/bcf/$1.chrM.bcf


#bcftools mpileup -f hg19.fa -O b --threads 7 bam/$1.bam > bcf/$1.bcf
#bcftools index bcf/$1.bcf
#bcftools call -r chrM -c -O b bcf/$1.bcf > chrM/bcf/$1.chrM.bcf
bcftools index  chrM/bcf/$1.chrM.bcf
samtools faidx hg19.fa chrM:1-16571 | bcftools consensus chrM/bcf/$1.chrM.bcf > chrM/fa/$1.chrM.fa





