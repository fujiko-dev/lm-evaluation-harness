export OPENAI_API_KEY="bce-v3/ALTAK-O0JXIpHCVwyiAn40hEsVe/3cc2daea23929dead2a3e65334415db7f184c57d"
./scripts/run_lca_batch.sh kimi-k2-instruct https://qianfan.baidubce.com/v2/chat/completions full



export HF_ALLOW_CODE_EVAL="1"
export OPENAI_API_KEY="bce-v3/ALTAK-O0JXIpHCVwyiAn40hEsVe/3cc2daea23929dead2a3e65334415db7f184c57d" 
lm_eval --model local-chat-completions --model_args model=kimi-k2-instruct,base_url=https://qianfan.baidubce.com/v2/chat/completions,num_concurrent=1,max_retries=10,tokenized_requests=False --tasks repobench_in_file_down_sampling_200 --apply_chat_template --output_path ./output --trust_remote_code --log_samples --confirm_run_unsafe_code


export OPENAI_API_KEY="bce-v3/ALTAK-O0JXIpHCVwyiAn40hEsVe/3cc2daea23929dead2a3e65334415db7f184c57d" && python3 -m lm_eval --model local-chat-completions --model_args "model=kimi-k2-instruct,base_url=https://qianfan.baidubce.com/v2/chat/completions,num_concurrent=1,max_retries=10,tokenized_requests=False" --apply_chat_template --tasks lca_library_based_code_generation_full --batch_size 1 --limit 3 --log_samples --output_path ./results/long_code_arena/library_based_code_generation/kimi_full_test_$(date +%Y-%m-%d_%H-%M-%S) --confirm_run_unsafe_code

export OPENAI_API_KEY="bce-v3/ALTAK-O0JXIpHCVwyiAn40hEsVe/3cc2daea23929dead2a3e65334415db7f184c57d" && python3 -m lm_eval --model local-chat-completions --model_args "model=kimi-k2-instruct,base_url=https://qianfan.baidubce.com/v2/chat/completions,num_concurrent=1,max_retries=10,tokenized_requests=False" --apply_chat_template --tasks lca_library_based_code_generation_full --batch_size 1 --log_samples --output_path ./results/long_code_arena/library_based_code_generation/kimi_full_150_$(date +%Y-%m-%d_%H-%M-%S) --confirm_run_unsafe_code


📈 进度详情:
==========
- 已完成: 6/150 样本 (4%)
- 当前速度: 约52秒/样本
- 预计总时间: ~2小时5分钟
- 预计完成时间: 约 15:40 左右


ps -ef | grep lm_eval | grep -v grep  
lsof -p 31076 | grep -E "(tty|pts)" | head -5



# 设置访问外网代理
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891

huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  --local-dir "/mnt/cfs_bj/zhangwenjing01/outputs/tmp/dataset/vcbench" \
cloudcatcher2/VCBench