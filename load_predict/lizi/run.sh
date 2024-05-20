#!/bin/bash

# 定义参数列表
# batchList=(64 128 256 512 1024)
# unitList=(4 8 16 32 64)
# futureList=(1 2 5)
# inputList=(5 10 15 20 25 30)
batchList=(1024)
unitList=(4 8 16 32 64)

# 遍历所有的batch和unit参数
for batch in "${batchList[@]}"; do
    for units1 in "${unitList[@]}"; do
        for units2 in "${unitList[@]}"; do
            for attention_units in "${unitList[@]}"; do
                # 为每次运行创建一个唯一的screen会话名
                session_name="python_session_${batch}_${units1}_${units2}_${attention_units}"
                echo "Starting screen session '$session_name'..."

                if ! screen -list | grep -q "$session_name"; then
                    # 创建新的screen会话，并在该会话中运行Python程序
                    screen -dmS $session_name

                    # 向新创建的screen会会话中发送Python程序运行命令，并附带相应的参数
                    screen -S $session_name -X stuff "conda activate lizi38 && CUDA_VISIBLE_DEVICES=-1 python compareWorkloadPredict.py --batch_size $batch --units1 $units1 --units2 $units2 --attention_units $attention_units; exit\n"
                else 
                    echo "Screen session '$session_name' already exists, skipping..."
                fi
            done
        done
    done
done

for futurestep in "${futureList[@]}"; do
    for inputstep in "${inputList[@]}"; do
                        # 为每次运行创建一个唯一的screen会话名
                        session_name="python_session_${futurestep}_${inputstep}_${batch}_${units1}_${units2}_${attention_units}"
                        echo "Starting screen session '$session_name'..."

                        if ! screen -list | grep -q "$session_name"; then
                            # 创建新的screen会话，并在该会话中运行Python程序
                            screen -dmS $session_name

                            # 向新创建的screen会会话中发送Python程序运行命令，并附带相应的参数
                            screen -S $session_name -X stuff "conda activate lizi38 && CUDA_VISIBLE_DEVICES=-1 python compareWorkloadPredict.py --future_step $futurestep --input_step $inputstep --batch_size $batch --units1 $units1 --units2 $units2 --attention_units $attention_units; exit\n"
                        else
                            echo "Screen session '$session_name' already exists, skipping..."
                        fi
    done
done