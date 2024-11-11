
# en2de
BASE_AUDIO_PATH=path/to/cv/audio
python speechqe/score_speechqe.py \
    --speechqe_model=h-j-han/SpeechQE-TowerInstruct-7B-en2de \
    --dataset_name=h-j-han/SpeechQE-CoVoST2 \
    --base_audio_path=$BASE_AUDIO_PATH \
    --dataset_config_name=en2de \
    --test_split_name=test  
    # --test_split_name=test_seamlar+test_seammid+test_tfw2vlg+test_tfmidmc+test_tfsmlmc  # to reproduce the numbers in the table


# es2en
BASE_AUDIO_PATH=path/to/cv/audio
python speechqe/score_speechqe.py \
    --speechqe_model=h-j-han/SpeechQE-TowerInstruct-7B-es2en \
    --dataset_name=h-j-han/SpeechQE-CoVoST2 \
    --base_audio_path=$BASE_AUDIO_PATH \
    --dataset_config_name=es2en \
    --test_split_name=test 
    --test_split_name=test_whsplv3+test_whsplv2+test_whspmid+test_whspsml+test_whspbas
