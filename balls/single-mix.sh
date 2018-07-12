work_dir=$1
ac_s=$2
hclg_s=$3
nnlm_s=$4
n_jobs=$5

echo "work_dir $work_dir"
echo "ac_s $ac_s"
echo "hclg_s $hclg_s"
echo "nnlm_s $nnlm_s"
echo "n_jobs $n_jobs"

pick_dir="pick-$ac_s-$hclg_s-$nnlm_s"

mkdir $work_dir/$pick_dir || { echo "directory $work_dir/$pick_dir already exists" >&2 ; exit 1 ; }

for ii in $(seq 1 $n_jobs) ; do 
    python rescoring_combine_scores.py \
        --ac-scale=$ac_s \
        --gr-scale=$hclg_s \
        --lm-scale=$nnlm_s \
        $work_dir/$ii.acscore \
        $work_dir/$ii.hclgscore \
        $work_dir/$ii.rnnlm-scores \
        $work_dir/$pick_dir/$ii.pick 

    lattice-copy ark:$work_dir/latt.$ii.nbest ark,t:- |\
        python ~/PhD/pyth-lm/balls/pick-best.py $work_dir/$pick_dir/$ii.pick |\
        lattice-copy ark,t:- ark:- |\
        gzip -c > $work_dir/$pick_dir/lat.$ii.gz 
done
