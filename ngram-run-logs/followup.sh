#!/bin/bash
# Follow-up experiments driven by the first battery's diagnosis: match
# precision, not lookup or snapshot overhead, is what costs W1 its speedup and
# what makes W2 regress. Also measures the snapshot copy strategy in situ.
set -u
cd /workspace
R=./ngram-run-logs/run.sh
P=./ngram-run-logs/prompts

echo "### W1 at min match 4, second repetition for best-of-2"
$R w1_ngram7_m4_r1 $P/w1.txt --speculative-ngram 7 --ngram-min-match 4
$R w1_ngram7_m4_r2 $P/w1.txt --speculative-ngram 7 --ngram-min-match 4
$R w1_ngram6_m4_r1 $P/w1.txt --speculative-ngram 6 --ngram-min-match 4
$R w1_ngram6_m4_r2 $P/w1.txt --speculative-ngram 6 --ngram-min-match 4

echo "### W2 and W3 at min match 4"
$R w2_ngram7_m4_r1 $P/w2.txt --speculative-ngram 7 --ngram-min-match 4
$R w3_ngram7_m4_r1 $P/w3.txt --speculative-ngram 7 --ngram-min-match 4
$R w2_ngram7_m4_r2 $P/w2.txt --speculative-ngram 7 --ngram-min-match 4
$R w3_ngram7_m4_r2 $P/w3.txt --speculative-ngram 7 --ngram-min-match 4

echo "### snapshot copy strategy, in situ on W1"
$R w1_ngram7_plaincopy $P/w1.txt --speculative-ngram 7 --snapshot-copy plain
$R w1_ngram7_streamcopy $P/w1.txt --speculative-ngram 7 --snapshot-copy streaming

echo "### done"
