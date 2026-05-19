#change
            attn_output_ip_adapter = hidden_states + self.scale * ip_hidden_states  # 保存原始 IP-Adapter 输出

            #  获取主体数量条件 (从 kwargs 中获取 num_subjects，如果不存在则使用默认值 1)
            num_subjects = kwargs.get("num_subjects", torch.tensor([1]).long().to(hidden_states.device))#torch.Size([1])

            #  获取主体数量条件 Embedding
            num_condition_embedding = self.ip_adapter.num_embeddings(num_subjects)  # torch.Size([1, 2048]) # 使用 self.ip_adapter.num_embedding

            #  构建数量控制 Cross-Attention 的 Key/Value
            num_control_key_value = num_condition_embedding.unsqueeze(0).unsqueeze(1)# torch.Size([1, 1, 1, 2048])


            #  矩阵乘法：将 self.wq.weight 乘到 q query.shape(torch.Size([2, 20, 3350, 64]))
            wq_weight = self.num_control_attn.wq.weight# torch.Size([1280, 1280])

            wq_weight_reshaped = wq_weight.reshape(wq_weight.shape[0], wq_weight.shape[1])#wq_weight shape: torch.Size([1280, 1280])
            #query_squeezed = query.squeeze(1).squeeze(1)
            #query_reshaped = query_squeezed.reshape(1280, -1).transpose(-1,-2)  # <-- NEW RESHAPE FOR QUERY IN attention_processor.py

            wq_weight_reshaped = wq_weight_reshaped.unsqueeze(0).repeat(1,2,1) #[2,1280,1280]

            query_reshaped = query.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).transpose(1, 2)#[2,1280,3350]

            #print("Shape of query_reshaped in attention_processor:",
            #      query_reshaped.shape)  # 打印 reshape 后的 query_reshaped 形状

            matmul_result = torch.matmul(wq_weight_reshaped, query_reshaped) #[2,1280,3350] # 使用 reshape 后的 query_reshaped 进行矩阵乘法

            print("Shape of matmul_result before num_control_attn:", matmul_result.shape)
            attn_output_num_control = self.num_control_attn(matmul_result)


            #  执行数量控制 Cross-Attention
            #attn_output_num_control = self.num_control_attn(
            #    query  # 使用与原始 IP-Adapter 相同的 query
                # key_value_states=num_control_key_value,
                # context=encoder_hidden_states  # context 可以继续沿用 encoder_hidden_states，或者根据需要修改
            #)

            #  融合原始 IP-Adapter 和数量控制 Cross-Attention 的输出 (例如，相加融合)
            hidden_states = attn_output_ip_adapter + attn_output_num_control  # 相加融合，并赋值给 hidden_states
        else:
            hidden_states = hidden_states

            # 以上change结束，这是被修改的原code    hidden_states = hidden_states + self.scale * ip_hidden_states