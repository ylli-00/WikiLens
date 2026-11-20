chunks: {'G': 43264, 'A': 124890, 'B': 59943, 'C': 12851, 'D': 72063, 'RA': 124904, 'RB': 59956, 'RD': 72066, 'AD': 72075}; claims with a reachable text-only evidence set: 2969

### article recall (claims hit; k = 1/5/10/20)

| group (n) | G | G+A | G+B | G+C | G+RA | G+RB | G+D | G+RD | G+AD |
|---|---|---|---|---|---|---|---|---|---|
| all (3000) | 2896/2962/2972/2976 | 2850/2944/2964/2970 | 2867/2957/2965/2973 | 2892/2960/2970/2976 | 2873/2953/2963/2972 | 2882/2956/2968/2973 | 2846/2952/2963/2971 | 2881/2955/2967/2972 | 2872/2952/2967/2972 |
| single (2100) | 2022/2068/2074/2078 | 1995/2058/2069/2073 | 2004/2065/2070/2075 | 2018/2067/2073/2078 | 2005/2061/2068/2075 | 2012/2063/2072/2076 | 1991/2062/2067/2073 | 2011/2062/2071/2075 | 2008/2062/2070/2074 |
| needs2 (900) | 874/894/898/898 | 855/886/895/897 | 863/892/895/898 | 874/893/897/898 | 868/892/895/897 | 870/893/896/897 | 855/890/896/898 | 870/893/896/897 | 864/890/897/898 |
| SUP single (700) | 696/697/698/698 | 692/697/698/698 | 694/697/698/698 | 694/697/698/698 | 696/697/698/698 | 696/697/698/698 | 695/697/698/698 | 696/697/698/698 | 694/697/698/698 |
| SUP needs2 (300) | 299/300/300/300 | 299/300/300/300 | 299/300/300/300 | 299/300/300/300 | 299/300/300/300 | 299/300/300/300 | 295/300/300/300 | 299/300/300/300 | 299/300/300/300 |
| REF single (700) | 681/697/697/698 | 668/692/696/698 | 673/696/697/698 | 680/697/697/698 | 675/695/696/698 | 677/695/697/698 | 668/695/695/698 | 676/695/697/698 | 675/694/697/698 |
| REF needs2 (300) | 292/297/299/299 | 285/293/298/299 | 288/297/298/299 | 292/297/298/299 | 288/297/298/298 | 290/297/298/298 | 285/297/298/299 | 290/297/298/298 | 289/295/298/299 |
| NEI single (700) | 645/674/679/682 | 635/669/675/677 | 637/672/675/679 | 644/673/678/682 | 634/669/674/679 | 639/671/677/680 | 628/670/674/677 | 639/670/676/679 | 639/671/675/678 |
| NEI needs2 (300) | 283/297/299/299 | 271/293/297/298 | 276/295/297/299 | 283/296/299/299 | 281/295/297/299 | 281/296/298/299 | 275/293/298/299 | 281/296/298/299 | 276/295/299/299 |

article drop from G in claims (k = 1/5/10/20), and per 10,000 added chunks

| group | G+A lost | G+A per 10k | G+B lost | G+B per 10k | G+C lost | G+C per 10k | G+RA lost | G+RA per 10k | G+RB lost | G+RB per 10k | G+D lost | G+D per 10k | G+RD lost | G+RD per 10k | G+AD lost | G+AD per 10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 46/18/8/6 | 3.7/1.4/0.6/0.5 | 29/5/7/3 | 4.8/0.8/1.2/0.5 | 4/2/2/0 | 3.1/1.6/1.6/0.0 | 23/9/9/4 | 1.8/0.7/0.7/0.3 | 14/6/4/3 | 2.3/1.0/0.7/0.5 | 50/10/9/5 | 6.9/1.4/1.2/0.7 | 15/7/5/4 | 2.1/1.0/0.7/0.6 | 24/10/5/4 | 3.3/1.4/0.7/0.6 |
| single | 27/10/5/5 | 2.2/0.8/0.4/0.4 | 18/3/4/3 | 3.0/0.5/0.7/0.5 | 4/1/1/0 | 3.1/0.8/0.8/0.0 | 17/7/6/3 | 1.4/0.6/0.5/0.2 | 10/5/2/2 | 1.7/0.8/0.3/0.3 | 31/6/7/5 | 4.3/0.8/1.0/0.7 | 11/6/3/3 | 1.5/0.8/0.4/0.4 | 14/6/4/4 | 1.9/0.8/0.6/0.6 |
| needs2 | 19/8/3/1 | 1.5/0.6/0.2/0.1 | 11/2/3/0 | 1.8/0.3/0.5/0.0 | 0/1/1/0 | 0.0/0.8/0.8/0.0 | 6/2/3/1 | 0.5/0.2/0.2/0.1 | 4/1/2/1 | 0.7/0.2/0.3/0.2 | 19/4/2/0 | 2.6/0.6/0.3/0.0 | 4/1/2/1 | 0.6/0.1/0.3/0.1 | 10/4/1/0 | 1.4/0.6/0.1/0.0 |
| SUP single | 4/0/0/0 | 0.3/0.0/0.0/0.0 | 2/0/0/0 | 0.3/0.0/0.0/0.0 | 2/0/0/0 | 1.6/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 1/0/0/0 | 0.1/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 2/0/0/0 | 0.3/0.0/0.0/0.0 |
| SUP needs2 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 4/0/0/0 | 0.6/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 |
| REF single | 13/5/1/0 | 1.0/0.4/0.1/0.0 | 8/1/0/0 | 1.3/0.2/0.0/0.0 | 1/0/0/0 | 0.8/0.0/0.0/0.0 | 6/2/1/0 | 0.5/0.2/0.1/0.0 | 4/2/0/0 | 0.7/0.3/0.0/0.0 | 13/2/2/0 | 1.8/0.3/0.3/0.0 | 5/2/0/0 | 0.7/0.3/0.0/0.0 | 6/3/0/0 | 0.8/0.4/0.0/0.0 |
| REF needs2 | 7/4/1/0 | 0.6/0.3/0.1/0.0 | 4/0/1/0 | 0.7/0.0/0.2/0.0 | 0/0/1/0 | 0.0/0.0/0.8/0.0 | 4/0/1/1 | 0.3/0.0/0.1/0.1 | 2/0/1/1 | 0.3/0.0/0.2/0.2 | 7/0/1/0 | 1.0/0.0/0.1/0.0 | 2/0/1/1 | 0.3/0.0/0.1/0.1 | 3/2/1/0 | 0.4/0.3/0.1/0.0 |
| NEI single | 10/5/4/5 | 0.8/0.4/0.3/0.4 | 8/2/4/3 | 1.3/0.3/0.7/0.5 | 1/1/1/0 | 0.8/0.8/0.8/0.0 | 11/5/5/3 | 0.9/0.4/0.4/0.2 | 6/3/2/2 | 1.0/0.5/0.3/0.3 | 17/4/5/5 | 2.4/0.6/0.7/0.7 | 6/4/3/3 | 0.8/0.6/0.4/0.4 | 6/3/4/4 | 0.8/0.4/0.6/0.6 |
| NEI needs2 | 12/4/2/1 | 1.0/0.3/0.2/0.1 | 7/2/2/0 | 1.2/0.3/0.3/0.0 | 0/1/0/0 | 0.0/0.8/0.0/0.0 | 2/2/2/0 | 0.2/0.2/0.2/0.0 | 2/1/1/0 | 0.3/0.2/0.2/0.0 | 8/4/1/0 | 1.1/0.6/0.1/0.0 | 2/1/1/0 | 0.3/0.1/0.1/0.0 | 7/2/0/0 | 1.0/0.3/0.0/0.0 |

### pageset recall (claims hit; k = 1/5/10/20)

| group (n) | G | G+A | G+B | G+C | G+RA | G+RB | G+D | G+RD | G+AD |
|---|---|---|---|---|---|---|---|---|---|
| all (3000) | 2022/2606/2753/2817 | 1995/2564/2691/2765 | 2004/2577/2714/2790 | 2018/2598/2750/2811 | 2005/2570/2715/2783 | 2012/2586/2730/2797 | 1991/2562/2695/2776 | 2011/2582/2727/2793 | 2008/2575/2708/2783 |
| single (2100) | 2022/2068/2074/2078 | 1995/2058/2069/2073 | 2004/2065/2070/2075 | 2018/2067/2073/2078 | 2005/2061/2068/2075 | 2012/2063/2072/2076 | 1991/2062/2067/2073 | 2011/2062/2071/2075 | 2008/2062/2070/2074 |
| needs2 (900) | 0/538/679/739 | 0/506/622/692 | 0/512/644/715 | 0/531/677/733 | 0/509/647/708 | 0/523/658/721 | 0/500/628/703 | 0/520/656/718 | 0/513/638/709 |
| SUP single (700) | 696/697/698/698 | 692/697/698/698 | 694/697/698/698 | 694/697/698/698 | 696/697/698/698 | 696/697/698/698 | 695/697/698/698 | 696/697/698/698 | 694/697/698/698 |
| SUP needs2 (300) | 0/240/273/282 | 0/226/261/277 | 0/228/266/280 | 0/237/273/282 | 0/235/268/279 | 0/238/271/280 | 0/226/262/277 | 0/238/270/280 | 0/229/267/279 |
| REF single (700) | 681/697/697/698 | 668/692/696/698 | 673/696/697/698 | 680/697/697/698 | 675/695/696/698 | 677/695/697/698 | 668/695/695/698 | 676/695/697/698 | 675/694/697/698 |
| REF needs2 (300) | 0/135/192/219 | 0/124/166/192 | 0/130/175/201 | 0/132/191/213 | 0/124/177/197 | 0/130/180/205 | 0/122/169/198 | 0/129/180/203 | 0/128/172/198 |
| NEI single (700) | 645/674/679/682 | 635/669/675/677 | 637/672/675/679 | 644/673/678/682 | 634/669/674/679 | 639/671/677/680 | 628/670/674/677 | 639/670/676/679 | 639/671/675/678 |
| NEI needs2 (300) | 0/163/214/238 | 0/156/195/223 | 0/154/203/234 | 0/162/213/238 | 0/150/202/232 | 0/155/207/236 | 0/152/197/228 | 0/153/206/235 | 0/156/199/232 |

pageset drop from G in claims (k = 1/5/10/20), and per 10,000 added chunks

| group | G+A lost | G+A per 10k | G+B lost | G+B per 10k | G+C lost | G+C per 10k | G+RA lost | G+RA per 10k | G+RB lost | G+RB per 10k | G+D lost | G+D per 10k | G+RD lost | G+RD per 10k | G+AD lost | G+AD per 10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 27/42/62/52 | 2.2/3.4/5.0/4.2 | 18/29/39/27 | 3.0/4.8/6.5/4.5 | 4/8/3/6 | 3.1/6.2/2.3/4.7 | 17/36/38/34 | 1.4/2.9/3.0/2.7 | 10/20/23/20 | 1.7/3.3/3.8/3.3 | 31/44/58/41 | 4.3/6.1/8.0/5.7 | 11/24/26/24 | 1.5/3.3/3.6/3.3 | 14/31/45/34 | 1.9/4.3/6.2/4.7 |
| single | 27/10/5/5 | 2.2/0.8/0.4/0.4 | 18/3/4/3 | 3.0/0.5/0.7/0.5 | 4/1/1/0 | 3.1/0.8/0.8/0.0 | 17/7/6/3 | 1.4/0.6/0.5/0.2 | 10/5/2/2 | 1.7/0.8/0.3/0.3 | 31/6/7/5 | 4.3/0.8/1.0/0.7 | 11/6/3/3 | 1.5/0.8/0.4/0.4 | 14/6/4/4 | 1.9/0.8/0.6/0.6 |
| needs2 | 0/32/57/47 | 0.0/2.6/4.6/3.8 | 0/26/35/24 | 0.0/4.3/5.8/4.0 | 0/7/2/6 | 0.0/5.4/1.6/4.7 | 0/29/32/31 | 0.0/2.3/2.6/2.5 | 0/15/21/18 | 0.0/2.5/3.5/3.0 | 0/38/51/36 | 0.0/5.3/7.1/5.0 | 0/18/23/21 | 0.0/2.5/3.2/2.9 | 0/25/41/30 | 0.0/3.5/5.7/4.2 |
| SUP single | 4/0/0/0 | 0.3/0.0/0.0/0.0 | 2/0/0/0 | 0.3/0.0/0.0/0.0 | 2/0/0/0 | 1.6/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 1/0/0/0 | 0.1/0.0/0.0/0.0 | 0/0/0/0 | 0.0/0.0/0.0/0.0 | 2/0/0/0 | 0.3/0.0/0.0/0.0 |
| SUP needs2 | 0/14/12/5 | 0.0/1.1/1.0/0.4 | 0/12/7/2 | 0.0/2.0/1.2/0.3 | 0/3/0/0 | 0.0/2.3/0.0/0.0 | 0/5/5/3 | 0.0/0.4/0.4/0.2 | 0/2/2/2 | 0.0/0.3/0.3/0.3 | 0/14/11/5 | 0.0/1.9/1.5/0.7 | 0/2/3/2 | 0.0/0.3/0.4/0.3 | 0/11/6/3 | 0.0/1.5/0.8/0.4 |
| REF single | 13/5/1/0 | 1.0/0.4/0.1/0.0 | 8/1/0/0 | 1.3/0.2/0.0/0.0 | 1/0/0/0 | 0.8/0.0/0.0/0.0 | 6/2/1/0 | 0.5/0.2/0.1/0.0 | 4/2/0/0 | 0.7/0.3/0.0/0.0 | 13/2/2/0 | 1.8/0.3/0.3/0.0 | 5/2/0/0 | 0.7/0.3/0.0/0.0 | 6/3/0/0 | 0.8/0.4/0.0/0.0 |
| REF needs2 | 0/11/26/27 | 0.0/0.9/2.1/2.2 | 0/5/17/18 | 0.0/0.8/2.8/3.0 | 0/3/1/6 | 0.0/2.3/0.8/4.7 | 0/11/15/22 | 0.0/0.9/1.2/1.8 | 0/5/12/14 | 0.0/0.8/2.0/2.3 | 0/13/23/21 | 0.0/1.8/3.2/2.9 | 0/6/12/16 | 0.0/0.8/1.7/2.2 | 0/7/20/21 | 0.0/1.0/2.8/2.9 |
| NEI single | 10/5/4/5 | 0.8/0.4/0.3/0.4 | 8/2/4/3 | 1.3/0.3/0.7/0.5 | 1/1/1/0 | 0.8/0.8/0.8/0.0 | 11/5/5/3 | 0.9/0.4/0.4/0.2 | 6/3/2/2 | 1.0/0.5/0.3/0.3 | 17/4/5/5 | 2.4/0.6/0.7/0.7 | 6/4/3/3 | 0.8/0.6/0.4/0.4 | 6/3/4/4 | 0.8/0.4/0.6/0.6 |
| NEI needs2 | 0/7/19/15 | 0.0/0.6/1.5/1.2 | 0/9/11/4 | 0.0/1.5/1.8/0.7 | 0/1/1/0 | 0.0/0.8/0.8/0.0 | 0/13/12/6 | 0.0/1.0/1.0/0.5 | 0/8/7/2 | 0.0/1.3/1.2/0.3 | 0/11/17/10 | 0.0/1.5/2.4/1.4 | 0/10/8/3 | 0.0/1.4/1.1/0.4 | 0/7/15/6 | 0.0/1.0/2.1/0.8 |

### evidence recall (claims hit; k = 1/5/10/20)

| group (n) | G | G+A | G+B | G+C | G+RA | G+RB | G+D | G+RD | G+AD |
|---|---|---|---|---|---|---|---|---|---|
| all (3000) | 1347/2351/2554/2659 | 1336/2293/2480/2603 | 1337/2307/2512/2627 | 1346/2341/2549/2657 | 1339/2302/2515/2625 | 1342/2321/2534/2642 | 1329/2290/2487/2612 | 1342/2315/2530/2640 | 1342/2310/2505/2621 |
| single (2100) | 1347/1962/2012/2039 | 1336/1935/1990/2026 | 1337/1943/2003/2034 | 1346/1957/2011/2039 | 1339/1936/2003/2033 | 1342/1945/2008/2036 | 1329/1935/1993/2028 | 1342/1942/2006/2035 | 1342/1943/1998/2029 |
| needs2 (900) | 0/389/542/620 | 0/358/490/577 | 0/364/509/593 | 0/384/538/618 | 0/366/512/592 | 0/376/526/606 | 0/355/494/584 | 0/373/524/605 | 0/367/507/592 |
| SUP single (700) | 460/686/690/692 | 460/680/688/692 | 458/681/690/692 | 460/684/690/692 | 460/680/689/692 | 460/682/689/692 | 460/682/690/692 | 460/681/689/692 | 460/681/689/692 |
| SUP needs2 (300) | 0/199/242/258 | 0/185/232/251 | 0/184/236/254 | 0/197/241/258 | 0/191/234/256 | 0/195/240/257 | 0/185/232/250 | 0/194/239/257 | 0/190/236/252 |
| REF single (700) | 519/671/680/688 | 511/663/674/686 | 514/667/678/687 | 518/671/680/688 | 515/666/677/687 | 516/667/679/688 | 510/664/675/686 | 516/667/678/688 | 516/666/678/687 |
| REF needs2 (300) | 0/88/141/173 | 0/82/116/149 | 0/86/124/158 | 0/87/138/171 | 0/83/128/156 | 0/86/130/165 | 0/80/121/154 | 0/85/130/165 | 0/84/122/157 |
| NEI single (700) | 368/605/642/659 | 365/592/628/648 | 365/595/635/655 | 368/602/641/659 | 364/590/637/654 | 366/596/640/656 | 359/589/628/650 | 366/594/639/655 | 366/596/631/650 |
| NEI needs2 (300) | 0/102/159/189 | 0/91/142/177 | 0/94/149/181 | 0/100/159/189 | 0/92/150/180 | 0/95/156/184 | 0/90/141/180 | 0/94/155/183 | 0/93/149/183 |

evidence drop from G in claims (k = 1/5/10/20), and per 10,000 added chunks

| group | G+A lost | G+A per 10k | G+B lost | G+B per 10k | G+C lost | G+C per 10k | G+RA lost | G+RA per 10k | G+RB lost | G+RB per 10k | G+D lost | G+D per 10k | G+RD lost | G+RD per 10k | G+AD lost | G+AD per 10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 11/58/74/56 | 0.9/4.6/5.9/4.5 | 10/44/42/32 | 1.7/7.3/7.0/5.3 | 1/10/5/2 | 0.8/7.8/3.9/1.6 | 8/49/39/34 | 0.6/3.9/3.1/2.7 | 5/30/20/17 | 0.8/5.0/3.3/2.8 | 18/61/67/47 | 2.5/8.5/9.3/6.5 | 5/36/24/19 | 0.7/5.0/3.3/2.6 | 5/41/49/38 | 0.7/5.7/6.8/5.3 |
| single | 11/27/22/13 | 0.9/2.2/1.8/1.0 | 10/19/9/5 | 1.7/3.2/1.5/0.8 | 1/5/1/0 | 0.8/3.9/0.8/0.0 | 8/26/9/6 | 0.6/2.1/0.7/0.5 | 5/17/4/3 | 0.8/2.8/0.7/0.5 | 18/27/19/11 | 2.5/3.7/2.6/1.5 | 5/20/6/4 | 0.7/2.8/0.8/0.6 | 5/19/14/10 | 0.7/2.6/1.9/1.4 |
| needs2 | 0/31/52/43 | 0.0/2.5/4.2/3.4 | 0/25/33/27 | 0.0/4.2/5.5/4.5 | 0/5/4/2 | 0.0/3.9/3.1/1.6 | 0/23/30/28 | 0.0/1.8/2.4/2.2 | 0/13/16/14 | 0.0/2.2/2.7/2.3 | 0/34/48/36 | 0.0/4.7/6.7/5.0 | 0/16/18/15 | 0.0/2.2/2.5/2.1 | 0/22/35/28 | 0.0/3.1/4.9/3.9 |
| SUP single | 0/6/2/0 | 0.0/0.5/0.2/0.0 | 2/5/0/0 | 0.3/0.8/0.0/0.0 | 0/2/0/0 | 0.0/1.6/0.0/0.0 | 0/6/1/0 | 0.0/0.5/0.1/0.0 | 0/4/1/0 | 0.0/0.7/0.2/0.0 | 0/4/0/0 | 0.0/0.6/0.0/0.0 | 0/5/1/0 | 0.0/0.7/0.1/0.0 | 0/5/1/0 | 0.0/0.7/0.1/0.0 |
| SUP needs2 | 0/14/10/7 | 0.0/1.1/0.8/0.6 | 0/15/6/4 | 0.0/2.5/1.0/0.7 | 0/2/1/0 | 0.0/1.6/0.8/0.0 | 0/8/8/2 | 0.0/0.6/0.6/0.2 | 0/4/2/1 | 0.0/0.7/0.3/0.2 | 0/14/10/8 | 0.0/1.9/1.4/1.1 | 0/5/3/1 | 0.0/0.7/0.4/0.1 | 0/9/6/6 | 0.0/1.2/0.8/0.8 |
| REF single | 8/8/6/2 | 0.6/0.6/0.5/0.2 | 5/4/2/1 | 0.8/0.7/0.3/0.2 | 1/0/0/0 | 0.8/0.0/0.0/0.0 | 4/5/3/1 | 0.3/0.4/0.2/0.1 | 3/4/1/0 | 0.5/0.7/0.2/0.0 | 9/7/5/2 | 1.2/1.0/0.7/0.3 | 3/4/2/0 | 0.4/0.6/0.3/0.0 | 3/5/2/1 | 0.4/0.7/0.3/0.1 |
| REF needs2 | 0/6/25/24 | 0.0/0.5/2.0/1.9 | 0/2/17/15 | 0.0/0.3/2.8/2.5 | 0/1/3/2 | 0.0/0.8/2.3/1.6 | 0/5/13/17 | 0.0/0.4/1.0/1.4 | 0/2/11/8 | 0.0/0.3/1.8/1.3 | 0/8/20/19 | 0.0/1.1/2.8/2.6 | 0/3/11/8 | 0.0/0.4/1.5/1.1 | 0/4/19/16 | 0.0/0.6/2.6/2.2 |
| NEI single | 3/13/14/11 | 0.2/1.0/1.1/0.9 | 3/10/7/4 | 0.5/1.7/1.2/0.7 | 0/3/1/0 | 0.0/2.3/0.8/0.0 | 4/15/5/5 | 0.3/1.2/0.4/0.4 | 2/9/2/3 | 0.3/1.5/0.3/0.5 | 9/16/14/9 | 1.2/2.2/1.9/1.2 | 2/11/3/4 | 0.3/1.5/0.4/0.6 | 2/9/11/9 | 0.3/1.2/1.5/1.2 |
| NEI needs2 | 0/11/17/12 | 0.0/0.9/1.4/1.0 | 0/8/10/8 | 0.0/1.3/1.7/1.3 | 0/2/0/0 | 0.0/1.6/0.0/0.0 | 0/10/9/9 | 0.0/0.8/0.7/0.7 | 0/7/3/5 | 0.0/1.2/0.5/0.8 | 0/12/18/9 | 0.0/1.7/2.5/1.2 | 0/8/4/6 | 0.0/1.1/0.6/0.8 | 0/9/10/6 | 0.0/1.2/1.4/0.8 |

### occupancy

- A: {"distractor_slots_in_top10": 10501, "slots_per_10k_added_chunks": 840.8199215309472, "claims_with_a_distractor_in_top10": 2493, "claims_with_a_distractor_at_rank1": 84, "slots_from_pages_linked_by_the_claims_own_gold_pages": 2761, "distinct_distractor_pages_in_any_top10": 2276, "distractor_pages_total": 3000, "share_of_slots_taken_by_top15_pages": 0.036187029806685075, "displaced_gold_page_chunks_from_top10": 1610}
- B: {"distractor_slots_in_top10": 8538, "slots_per_10k_added_chunks": 1424.3531354787049, "claims_with_a_distractor_in_top10": 2340, "claims_with_a_distractor_at_rank1": 57, "slots_from_pages_linked_by_the_claims_own_gold_pages": 2452, "distinct_distractor_pages_in_any_top10": 2025, "distractor_pages_total": 3000, "share_of_slots_taken_by_top15_pages": 0.04275005856172406, "displaced_gold_page_chunks_from_top10": 1044}
- C: {"distractor_slots_in_top10": 2675, "slots_per_10k_added_chunks": 2081.5500739242084, "claims_with_a_distractor_in_top10": 1322, "claims_with_a_distractor_at_rank1": 11, "slots_from_pages_linked_by_the_claims_own_gold_pages": 25, "distinct_distractor_pages_in_any_top10": 1193, "distractor_pages_total": 3000, "share_of_slots_taken_by_top15_pages": 0.06542056074766354, "displaced_gold_page_chunks_from_top10": 148}
- RA: {"distractor_slots_in_top10": 9250, "slots_per_10k_added_chunks": 740.5687568052265, "claims_with_a_distractor_in_top10": 2319, "claims_with_a_distractor_at_rank1": 48, "slots_from_pages_linked_by_the_claims_own_gold_pages": 125, "distinct_distractor_pages_in_any_top10": 5600, "distractor_pages_total": 29750, "share_of_slots_taken_by_top15_pages": 0.017945945945945944, "displaced_gold_page_chunks_from_top10": 894}
- RB: {"distractor_slots_in_top10": 6867, "slots_per_10k_added_chunks": 1145.3399159383548, "claims_with_a_distractor_in_top10": 2060, "claims_with_a_distractor_at_rank1": 34, "slots_from_pages_linked_by_the_claims_own_gold_pages": 77, "distinct_distractor_pages_in_any_top10": 3749, "distractor_pages_total": 14311, "share_of_slots_taken_by_top15_pages": 0.027377311780981507, "displaced_gold_page_chunks_from_top10": 529}
- D: {"distractor_slots_in_top10": 9892, "slots_per_10k_added_chunks": 1372.6877870751982, "claims_with_a_distractor_in_top10": 2513, "claims_with_a_distractor_at_rank1": 81, "slots_from_pages_linked_by_the_claims_own_gold_pages": 3873, "distinct_distractor_pages_in_any_top10": 2218, "distractor_pages_total": 3000, "share_of_slots_taken_by_top15_pages": 0.03427011726647796, "displaced_gold_page_chunks_from_top10": 1496}
- RD: {"distractor_slots_in_top10": 7505, "slots_per_10k_added_chunks": 1041.4064884966558, "claims_with_a_distractor_in_top10": 2127, "claims_with_a_distractor_at_rank1": 36, "slots_from_pages_linked_by_the_claims_own_gold_pages": 84, "distinct_distractor_pages_in_any_top10": 4193, "distractor_pages_total": 17155, "share_of_slots_taken_by_top15_pages": 0.023717521652231847, "displaced_gold_page_chunks_from_top10": 602}
- AD: {"distractor_slots_in_top10": 8486, "slots_per_10k_added_chunks": 1177.3846687478322, "claims_with_a_distractor_in_top10": 2311, "claims_with_a_distractor_at_rank1": 53, "slots_from_pages_linked_by_the_claims_own_gold_pages": 2275, "distinct_distractor_pages_in_any_top10": 1315, "distractor_pages_total": 1573, "share_of_slots_taken_by_top15_pages": 0.05267499410794249, "displaced_gold_page_chunks_from_top10": 1159}

### top offenders (top-10 lists)


A:

| page | n_chunks | claims | slots | displaced_gold_chunks | lost_article10 | lost_evidence10 | linked_from_claim_gold | gold_pages_linking |
|---|---|---|---|---|---|---|---|---|
| Double (association football) | 240 | 20 | 25 | 3.8 | 0 | 1 | 0 | 7 |
| Moscow | 98 | 18 | 28 | 2.1 | 0 | 0 | 5 | 30 |
| University of Cambridge | 69 | 18 | 25 | 4.4 | 0 | 0 | 2 | 19 |
| University of Oxford | 60 | 17 | 26 | 3.5 | 0 | 0 | 3 | 18 |
| Russia | 94 | 17 | 22 | 3.0 | 0 | 0 | 4 | 122 |
| University of California, Berkeley | 70 | 16 | 28 | 3.3 | 0 | 0 | 2 | 19 |
| Shenyang | 72 | 16 | 19 | 1.0 | 0 | 1 | 3 | 9 |
| Oslo | 56 | 15 | 30 | 1.6 | 0 | 0 | 6 | 17 |
| England | 87 | 15 | 19 | 1.5 | 0 | 0 | 6 | 155 |
| Paris | 104 | 15 | 17 | 1.4 | 0 | 0 | 2 | 60 |
| Asparagales | 33 | 14 | 30 | 2.6 | 1 | 0 | 1 | 6 |
| Lepidoptera | 70 | 14 | 29 | 2.4 | 0 | 0 | 11 | 32 |
| Michigan Wolverines football | 56 | 13 | 42 | 7.6 | 0 | 0 | 4 | 14 |
| Poland | 101 | 13 | 20 | 1.5 | 0 | 0 | 5 | 69 |
| Yorkshire | 59 | 13 | 20 | 5.5 | 0 | 0 | 3 | 18 |

B:

| page | n_chunks | claims | slots | displaced_gold_chunks | lost_article10 | lost_evidence10 | linked_from_claim_gold |
|---|---|---|---|---|---|---|---|
| Russia | 94 | 24 | 32 | 3.8 | 0 | 0 | 5 |
| England | 87 | 23 | 27 | 3.1 | 1 | 2 | 7 |
| Moscow | 98 | 22 | 37 | 3.2 | 0 | 0 | 5 |
| Poland | 101 | 19 | 27 | 2.8 | 0 | 0 | 6 |
| Germany | 49 | 18 | 21 | 2.5 | 0 | 0 | 5 |
| University of California, Berkeley | 70 | 17 | 35 | 3.8 | 0 | 0 | 2 |
| British literature | 73 | 17 | 24 | 0.7 | 0 | 0 | 0 |
| Fungus | 59 | 16 | 22 | 1.0 | 0 | 0 | 1 |
| Mexico | 119 | 16 | 19 | 5.5 | 0 | 1 | 3 |
| Nazi Germany | 88 | 16 | 17 | 3.7 | 0 | 1 | 4 |
| Adelaide | 65 | 15 | 24 | 3.6 | 0 | 0 | 4 |
| Nigeria | 88 | 15 | 23 | 1.0 | 0 | 0 | 7 |
| Feyenoord | 57 | 15 | 20 | 2.2 | 0 | 1 | 1 |
| Birmingham | 74 | 15 | 19 | 3.9 | 0 | 2 | 2 |
| Leipzig | 74 | 15 | 18 | 1.6 | 0 | 2 | 2 |

C:

| page | n_chunks | claims | slots | displaced_gold_chunks | lost_article10 | lost_evidence10 | linked_from_claim_gold |
|---|---|---|---|---|---|---|---|
| December 1909 | 32 | 16 | 16 | 0.0 | 0 | 0 | 0 |
| Chrysotoxum cautum | 4 | 13 | 15 | 0.0 | 0 | 0 | 0 |
| History of Azerbaijan | 60 | 10 | 18 | 4.0 | 0 | 0 | 1 |
| Evgeny Kuznetsov | 11 | 10 | 11 | 2.1 | 0 | 1 | 1 |
| Afghanistan national football team | 28 | 9 | 13 | 2.0 | 0 | 0 | 0 |
| Letalnica bratov Gorišek | 17 | 9 | 13 | 0.7 | 0 | 0 | 0 |
| Scenes of Clerical Life | 24 | 9 | 13 | 0.0 | 0 | 0 | 0 |
| Foreign policy of the Bill Clinton administration | 57 | 9 | 11 | 0.0 | 0 | 0 | 0 |
| March 1930 | 32 | 9 | 10 | 0.0 | 0 | 0 | 0 |
| Tony Shillitoe | 9 | 9 | 10 | 0.0 | 0 | 0 | 0 |
| Jonathan Spector | 18 | 9 | 9 | 0.0 | 0 | 0 | 0 |
| Ulrich von Wilamowitz-Moellendorff | 13 | 8 | 10 | 0.0 | 0 | 0 | 0 |
| Crystal Gayle | 31 | 8 | 9 | 0.0 | 0 | 0 | 0 |
| Young Folks | 10 | 8 | 9 | 1.0 | 0 | 0 | 0 |
| Northern Woods and Water Route | 20 | 8 | 8 | 0.0 | 0 | 0 | 0 |

RA:

| page | n_chunks | claims | slots | displaced_gold_chunks | lost_article10 | lost_evidence10 | linked_from_claim_gold |
|---|---|---|---|---|---|---|---|
| Postcolonial literature | 58 | 12 | 13 | 1.4 | 0 | 0 | 0 |
| Gay literature | 38 | 11 | 12 | 1.2 | 0 | 0 | 0 |
| Xi'an | 49 | 10 | 11 | 1.2 | 0 | 0 | 1 |
| History of education in England | 44 | 9 | 18 | 0.0 | 0 | 0 | 0 |
| Königsberg | 40 | 9 | 13 | 1.4 | 0 | 0 | 0 |
| Homoisocitrate dehydrogenase | 2 | 9 | 9 | 0.5 | 0 | 0 | 0 |
| John V. Tucker | 7 | 9 | 9 | 0.9 | 0 | 0 | 0 |
| Nahuas of La Huasteca | 27 | 8 | 13 | 2.5 | 0 | 0 | 0 |
| German Americans | 81 | 8 | 11 | 1.0 | 0 | 1 | 0 |
| 2018 United States Senate elections | 49 | 8 | 10 | 2.7 | 1 | 1 | 0 |
| Thiosulfate dehydrogenase (quinone) | 1 | 8 | 8 | 0.5 | 0 | 0 | 0 |
| September 2012 in sports | 82 | 7 | 12 | 1.2 | 0 | 0 | 0 |
| KREH | 7 | 7 | 10 | 0.1 | 0 | 0 | 0 |
| Michael Tendler | 5 | 7 | 9 | 0.0 | 0 | 0 | 0 |
| Fevansia | 5 | 7 | 8 | 1.2 | 0 | 0 | 0 |

RB:

| page | n_chunks | claims | slots | displaced_gold_chunks | lost_article10 | lost_evidence10 | linked_from_claim_gold |
|---|---|---|---|---|---|---|---|
| Gay literature | 38 | 16 | 17 | 1.2 | 0 | 0 | 0 |
| Postcolonial literature | 58 | 16 | 17 | 1.2 | 0 | 0 | 0 |
| Xi'an | 49 | 13 | 16 | 1.5 | 0 | 0 | 1 |
| September 2012 in sports | 82 | 12 | 19 | 1.5 | 0 | 0 | 0 |
| 1010s | 55 | 10 | 10 | 1.0 | 0 | 0 | 0 |
| Appias albina | 8 | 9 | 15 | 2.6 | 0 | 0 | 0 |
| Nahuas of La Huasteca | 27 | 9 | 15 | 2.5 | 0 | 0 | 0 |
| Fevansia | 5 | 9 | 11 | 1.2 | 0 | 0 | 0 |
| 730s | 50 | 9 | 10 | 1.5 | 0 | 0 | 0 |
| Index of biochemistry articles | 28 | 9 | 10 | 0.3 | 0 | 0 | 0 |
| KPBA (1270 AM) | 7 | 9 | 10 | 0.3 | 0 | 0 | 0 |
| Khyber Pakhtunkhwa | 54 | 9 | 9 | 0.5 | 0 | 0 | 1 |
| Thiosulfate dehydrogenase (quinone) | 1 | 9 | 9 | 0.5 | 0 | 0 | 0 |
| Trans-Siberian Orchestra | 28 | 8 | 11 | 0.0 | 0 | 0 | 0 |
| Silvester Horne | 7 | 8 | 9 | 0.5 | 0 | 0 | 0 |

D:

| page | n_chunks | claims | slots | displaced_gold_chunks | lost_article10 | lost_evidence10 | linked_from_claim_gold |
|---|---|---|---|---|---|---|---|
| University of Cambridge | 69 | 24 | 33 | 5.3 | 0 | 0 | 4 |
| England | 87 | 23 | 28 | 2.6 | 1 | 1 | 7 |
| Moscow | 98 | 19 | 29 | 2.0 | 0 | 0 | 5 |
| Russia | 94 | 19 | 24 | 3.5 | 0 | 0 | 4 |
| Germany | 49 | 17 | 21 | 1.5 | 0 | 0 | 7 |
| Iran | 98 | 16 | 24 | 1.0 | 0 | 0 | 2 |
| Benjamin Netanyahu | 88 | 15 | 19 | 1.0 | 0 | 0 | 1 |
| Australia | 52 | 15 | 17 | 4.7 | 0 | 0 | 4 |
| Duke University | 64 | 14 | 23 | 2.9 | 0 | 0 | 2 |
| Poland | 101 | 14 | 21 | 2.5 | 0 | 0 | 6 |
| City of London | 60 | 14 | 20 | 2.6 | 0 | 1 | 1 |
| Hungary | 89 | 14 | 20 | 0.3 | 0 | 0 | 4 |
| France | 118 | 14 | 18 | 5.0 | 0 | 0 | 2 |
| Rome | 79 | 14 | 16 | 1.5 | 0 | 0 | 4 |
| Oslo | 56 | 13 | 26 | 1.8 | 0 | 0 | 5 |

RD:

| page | n_chunks | claims | slots | displaced_gold_chunks | lost_article10 | lost_evidence10 | linked_from_claim_gold |
|---|---|---|---|---|---|---|---|
| Gay literature | 38 | 15 | 16 | 1.2 | 0 | 0 | 0 |
| Postcolonial literature | 58 | 13 | 14 | 1.4 | 0 | 0 | 0 |
| Xi'an | 49 | 12 | 15 | 1.5 | 0 | 0 | 1 |
| September 2012 in sports | 82 | 11 | 18 | 1.5 | 0 | 0 | 0 |
| 1010s | 55 | 10 | 10 | 1.0 | 0 | 0 | 0 |
| Nahuas of La Huasteca | 27 | 9 | 15 | 2.5 | 0 | 0 | 0 |
| 730s | 50 | 9 | 10 | 1.5 | 0 | 0 | 0 |
| KPBA (1270 AM) | 7 | 9 | 10 | 0.3 | 0 | 0 | 0 |
| Khyber Pakhtunkhwa | 54 | 9 | 9 | 0.3 | 0 | 0 | 1 |
| Thiosulfate dehydrogenase (quinone) | 1 | 9 | 9 | 0.5 | 0 | 0 | 0 |
| Appias albina | 8 | 8 | 14 | 2.6 | 0 | 0 | 0 |
| Trans-Siberian Orchestra | 28 | 8 | 11 | 0.0 | 0 | 0 | 0 |
| Fevansia | 5 | 8 | 10 | 1.2 | 0 | 0 | 0 |
| Silvester Horne | 7 | 8 | 9 | 0.5 | 0 | 0 | 0 |
| Becontree | 21 | 8 | 8 | 1.3 | 0 | 0 | 1 |

AD:

| page | n_chunks | claims | slots | displaced_gold_chunks | lost_article10 | lost_evidence10 | linked_from_claim_gold |
|---|---|---|---|---|---|---|---|
| University of Cambridge | 69 | 24 | 34 | 5.0 | 0 | 0 | 4 |
| England | 87 | 23 | 29 | 3.0 | 0 | 1 | 8 |
| Moscow | 98 | 22 | 32 | 2.2 | 0 | 0 | 6 |
| Germany | 49 | 22 | 25 | 2.9 | 0 | 0 | 10 |
| University of Oxford | 60 | 21 | 33 | 3.6 | 0 | 0 | 3 |
| University of California, Berkeley | 70 | 20 | 34 | 3.3 | 0 | 0 | 2 |
| Russia | 94 | 19 | 25 | 3.7 | 0 | 0 | 4 |
| Nazi Germany | 88 | 19 | 21 | 3.5 | 0 | 1 | 4 |
| Oslo | 56 | 18 | 37 | 1.5 | 0 | 0 | 6 |
| Yorkshire | 59 | 18 | 26 | 5.7 | 0 | 0 | 4 |
| Lepidoptera | 70 | 17 | 32 | 2.3 | 0 | 0 | 13 |
| Pittsburgh | 82 | 17 | 21 | 3.1 | 0 | 0 | 2 |
| Michigan Wolverines football | 56 | 16 | 51 | 7.6 | 0 | 0 | 4 |
| City of London | 60 | 16 | 24 | 3.6 | 0 | 1 | 2 |
| Poland | 101 | 16 | 23 | 1.7 | 0 | 0 | 6 |

### size buckets

- A: [{"chunks_per_page": "0-4", "pages": 85, "chunks": 256, "top10_slots": 39, "slots_per_10k_chunks": 1523.4375}, {"chunks_per_page": "5-19", "pages": 596, "chunks": 7403, "top10_slots": 711, "slots_per_10k_chunks": 960.4214507632041}, {"chunks_per_page": "20-49", "pages": 1272, "chunks": 44382, "top10_slots": 3928, "slots_per_10k_chunks": 885.0434860979676}, {"chunks_per_page": "50-99", "pages": 987, "chunks": 65936, "top10_slots": 5419, "slots_per_10k_chunks": 821.8575588449405}, {"chunks_per_page": "100-max", "pages": 60, "chunks": 6913, "top10_slots": 404, "slots_per_10k_chunks": 584.4061912339071}]
- B: [{"chunks_per_page": "0-4", "pages": 659, "chunks": 1676, "top10_slots": 528, "slots_per_10k_chunks": 3150.3579952267305}, {"chunks_per_page": "5-19", "pages": 1333, "chunks": 13373, "top10_slots": 2219, "slots_per_10k_chunks": 1659.3135422119196}, {"chunks_per_page": "20-49", "pages": 676, "chunks": 21535, "top10_slots": 2807, "slots_per_10k_chunks": 1303.4594845600184}, {"chunks_per_page": "50-99", "pages": 310, "chunks": 20804, "top10_slots": 2753, "slots_per_10k_chunks": 1323.3032109209767}, {"chunks_per_page": "100-max", "pages": 22, "chunks": 2555, "top10_slots": 231, "slots_per_10k_chunks": 904.1095890410959}]
- C: [{"chunks_per_page": "0-4", "pages": 2130, "chunks": 4268, "top10_slots": 1031, "slots_per_10k_chunks": 2415.651358950328}, {"chunks_per_page": "5-19", "pages": 791, "chunks": 6224, "top10_slots": 1312, "slots_per_10k_chunks": 2107.9691516709513}, {"chunks_per_page": "20-49", "pages": 72, "chunks": 1956, "top10_slots": 277, "slots_per_10k_chunks": 1416.155419222904}, {"chunks_per_page": "50-99", "pages": 7, "chunks": 403, "top10_slots": 55, "slots_per_10k_chunks": 1364.7642679900744}, {"chunks_per_page": "100-max", "pages": 0, "chunks": 0, "top10_slots": 0, "slots_per_10k_chunks": null}]
- RA: [{"chunks_per_page": "0-4", "pages": 21456, "chunks": 42585, "top10_slots": 3422, "slots_per_10k_chunks": 803.5693319243866}, {"chunks_per_page": "5-19", "pages": 7653, "chunks": 60798, "top10_slots": 4614, "slots_per_10k_chunks": 758.9065429783875}, {"chunks_per_page": "20-49", "pages": 558, "chunks": 15770, "top10_slots": 987, "slots_per_10k_chunks": 625.8719086873812}, {"chunks_per_page": "50-99", "pages": 77, "chunks": 4814, "top10_slots": 210, "slots_per_10k_chunks": 436.22766929788116}, {"chunks_per_page": "100-max", "pages": 6, "chunks": 937, "top10_slots": 17, "slots_per_10k_chunks": 181.4300960512273}]
- RB: [{"chunks_per_page": "0-4", "pages": 10304, "chunks": 20525, "top10_slots": 2585, "slots_per_10k_chunks": 1259.4397076735686}, {"chunks_per_page": "5-19", "pages": 3706, "chunks": 29553, "top10_slots": 3430, "slots_per_10k_chunks": 1160.6266707271682}, {"chunks_per_page": "20-49", "pages": 264, "chunks": 7269, "top10_slots": 676, "slots_per_10k_chunks": 929.9766130141697}, {"chunks_per_page": "50-99", "pages": 35, "chunks": 2171, "top10_slots": 162, "slots_per_10k_chunks": 746.1999078765547}, {"chunks_per_page": "100-max", "pages": 2, "chunks": 438, "top10_slots": 14, "slots_per_10k_chunks": 319.63470319634706}]
- D: [{"chunks_per_page": "0-4", "pages": 501, "chunks": 1303, "top10_slots": 527, "slots_per_10k_chunks": 4044.5126630851883}, {"chunks_per_page": "5-19", "pages": 1209, "chunks": 13003, "top10_slots": 2503, "slots_per_10k_chunks": 1924.940398369607}, {"chunks_per_page": "20-49", "pages": 845, "chunks": 27002, "top10_slots": 3538, "slots_per_10k_chunks": 1310.2733130879194}, {"chunks_per_page": "50-99", "pages": 421, "chunks": 27940, "top10_slots": 3064, "slots_per_10k_chunks": 1096.6356478167502}, {"chunks_per_page": "100-max", "pages": 24, "chunks": 2815, "top10_slots": 260, "slots_per_10k_chunks": 923.6234458259325}]
- RD: [{"chunks_per_page": "0-4", "pages": 12343, "chunks": 24511, "top10_slots": 2785, "slots_per_10k_chunks": 1136.2245522418507}, {"chunks_per_page": "5-19", "pages": 4445, "chunks": 35469, "top10_slots": 3786, "slots_per_10k_chunks": 1067.410978601032}, {"chunks_per_page": "20-49", "pages": 325, "chunks": 9099, "top10_slots": 740, "slots_per_10k_chunks": 813.2761841960655}, {"chunks_per_page": "50-99", "pages": 39, "chunks": 2409, "top10_slots": 171, "slots_per_10k_chunks": 709.8381070983811}, {"chunks_per_page": "100-max", "pages": 3, "chunks": 578, "top10_slots": 23, "slots_per_10k_chunks": 397.92387543252596}]
- AD: [{"chunks_per_page": "0-4", "pages": 44, "chunks": 131, "top10_slots": 23, "slots_per_10k_chunks": 1755.7251908396945}, {"chunks_per_page": "5-19", "pages": 267, "chunks": 3259, "top10_slots": 387, "slots_per_10k_chunks": 1187.4808223381403}, {"chunks_per_page": "20-49", "pages": 576, "chunks": 20305, "top10_slots": 2618, "slots_per_10k_chunks": 1289.3376015759666}, {"chunks_per_page": "50-99", "pages": 641, "chunks": 43288, "top10_slots": 5038, "slots_per_10k_chunks": 1163.8329329144335}, {"chunks_per_page": "100-max", "pages": 45, "chunks": 5092, "top10_slots": 420, "slots_per_10k_chunks": 824.8232521602514}]

sanity: {"A_in_gold": 0, "B_in_gold": 0, "C_in_gold": 0, "parsed_title_mismatch": 0, "parsed_title_of_distractor_in_gold": 0, "gold_pages_without_chunks": 0, "A_and_B_overlap": 585, "group_sizes": {"all": 3000, "single": 2100, "needs2": 900, "SUP single": 700, "SUP needs2": 300, "REF single": 700, "REF needs2": 300, "NEI single": 700, "NEI needs2": 300}, "c_skip": {"gold": 3, "in B": 4, "in A": 2}, "c_zero_chunk_pages": 1, "numpy_crosscheck_mismatch": 0}

search (GPU, 4 blocks): 1.0s; evaluate total 15.8s

### exact McNemar (first hits & second misses / reverse / p)

- G+D vs G+RD pageset@5: 12 / 32, p = 0.0037
- G+D vs G+RD pageset@10: 7 / 39, p = 1.8e-06
- G+D vs G+RD evidence@5: 17 / 42, p = 0.0015
- G+D vs G+RD evidence@10: 7 / 50, p = 4.2e-09
- G+D vs G+AD pageset@5: 9 / 22, p = 0.029
- G+D vs G+AD pageset@10: 8 / 21, p = 0.024
- G+D vs G+AD evidence@5: 12 / 32, p = 0.0037
- G+D vs G+AD evidence@10: 15 / 33, p = 0.013
- G+D vs G+A pageset@5: 16 / 18, p = 0.86
- G+D vs G+A pageset@10: 18 / 14, p = 0.6
- G+D vs G+A evidence@5: 24 / 27, p = 0.78
- G+D vs G+A evidence@10: 31 / 24, p = 0.42
- G+D vs G+B pageset@5: 12 / 27, p = 0.024
- G+D vs G+B pageset@10: 5 / 24, p = 0.00055
- G+D vs G+B evidence@5: 15 / 32, p = 0.019
- G+D vs G+B evidence@10: 9 / 34, p = 0.00017
- G+A vs G+RA pageset@5: 16 / 22, p = 0.42
- G+A vs G+RA pageset@10: 11 / 35, p = 0.00054
- G+A vs G+RA evidence@5: 19 / 28, p = 0.24
- G+A vs G+RA evidence@10: 12 / 47, p = 5.1e-06
- G+B vs G+RB pageset@5: 10 / 19, p = 0.14
- G+B vs G+RB pageset@10: 8 / 24, p = 0.007
- G+B vs G+RB evidence@5: 13 / 27, p = 0.038
- G+B vs G+RB evidence@10: 7 / 29, p = 0.00031
