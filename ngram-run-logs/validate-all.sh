#!/bin/bash
# Full n-gram speculation validation battery. Sequential by design: every run
# gets the whole machine, 6 physical cores, no thread env vars.
set -u
cd /workspace
R=./ngram-run-logs/run.sh
P=./ngram-run-logs/prompts

echo "### core workloads: greedy equivalence + best-of-2 timing"
$R w2_base_r1   $P/w2.txt --speculative-ngram 0
$R w2_ngram7_r1 $P/w2.txt --speculative-ngram 7
$R w3_base_r1   $P/w3.txt --speculative-ngram 0
$R w3_ngram7_r1 $P/w3.txt --speculative-ngram 7

echo "### second repetitions"
$R w1_base_r2   $P/w1.txt --speculative-ngram 0
$R w1_ngram7_r2 $P/w1.txt --speculative-ngram 7
$R w2_base_r2   $P/w2.txt --speculative-ngram 0
$R w2_ngram7_r2 $P/w2.txt --speculative-ngram 7
$R w3_base_r2   $P/w3.txt --speculative-ngram 0
$R w3_ngram7_r2 $P/w3.txt --speculative-ngram 7

echo "### W1 sweep: draft length x minimum match length"
for mm in 2 3; do
  for dl in 4 6 7 8 12; do
    $R "sweep_w1_d${dl}_m${mm}" $P/w1.txt --speculative-ngram "$dl" --ngram-min-match "$mm"
  done
done
echo "### W1 sweep: min match 4 (extra diagnostic cells)"
for dl in 6 7 8; do
  $R "sweep_w1_d${dl}_m4" $P/w1.txt --speculative-ngram "$dl" --ngram-min-match 4
done

echo "### MTP non-regression smoke"
$R mtp_draft1 $P/w1.txt --speculative-mtp 1

echo "### done"
