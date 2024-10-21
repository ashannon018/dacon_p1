research
https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations


(base) ➜  reference_data git:(main) ✗ grep -rn 'AHNAK*' .
./BRCA.csv:155:AHNAK
./PAAD.csv:475:AHNAK2
./KIPAN.csv:427:AHNAK2
./KIPAN.csv:618:AHNAK
./PRAD.csv:24:AHNAK

(base) ➜  reference_data git:(main) ✗ grep -rn 'CACNA1*' . 
./KIPAN.csv:208:CACNA2D2


(base) ➜  reference_data git:(main) ✗ grep -rn 'DOCK4*' . 
./GBMLGG.csv:164:DOCK6

(base) ➜  reference_data git:(main) ✗ grep -rn 'POLR2A*' .
./COAD.csv:343:POLR2D

qq = ['AHNAK', 'TCHH', 'APC', 'CACNA1A', 'CPEB2', 'MAGEE1', 'TP53', 'KDM6B', 'TUBGCP6', 'KMT2D', 'HLA-A', 'UBC']
{'AHNAK', 'APC', 'CACNA1A', 'DOCK4', 'POLR2A', 'STAB1', 'TCHH', 'TP53'}
for q in qq:
    print(q)
    explore_genes_distribution(q)

2519 43
2524 46
2525 42

{'AHNAK', 'APC', 'FBXW7', 'TCHH', 'TP53', 'TUBGCP6'}
{'AHNAK', 'APC', 'CEL', 'CES1', 'MAGEE1', 'TP53', 'TUBGCP6'}


(base) ➜  reference_data git:(main) ✗ grep -rn 'HLA-A*' .
./GBMLGG.csv:33:HLA-C
./GBMLGG.csv:139:HLA-B
./GBMLGG.csv:168:HLA-A
./PAAD.csv:278:HLA-A
./PAAD.csv:295:HLA-C
./PAAD.csv:394:HLA-B
./PAAD.csv:415:HLA-DQB2
./LUSC.csv:15:HLA-DPA1
./LUSC.csv:25:HLA-DQA1
./LUSC.csv:39:HLA-G
./LUSC.csv:44:HLA-DQA2
./LUSC.csv:46:HLA-C
./LUSC.csv:69:HLA-DRA
./LUSC.csv:74:HLA-DQB1
./LUSC.csv:122:HLA-DRB5
./CESC.csv:4:HLA-DPB1
./CESC.csv:31:HLA-DQB1
./LAML.csv:81:HLA-DQA2
./LAML.csv:97:HLA-DQB1
./LAML.csv:106:HLA-DMB
./LAML.csv:167:HLA-DRB5
./LAML.csv:347:HLA-G
./LAML.csv:349:HLA-B
./LAML.csv:632:HLA-B
./LAML.csv:724:HLA-DQA1
./LAML.csv:986:HLA-A
BRCA      539
PAAD      472 2.32
COAD      286
KIPAN     193
STES      159
GBMLGG    122
OV        104
THCA       69
KIRC       66
LGG        63
HNSC       57
UCEC       55
SKCM       52
CESC       41
PRAD       40
PCPG       35
TGCT       34
LAML       32
LIHC       29
SARC       29
LUAD       24
BLCA       21
ACC        12
LUSC        7
THYM        5