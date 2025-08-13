import torch
# from peft import LoraConfig, get_peft_model

def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(args, config=None):
    
    from transformers import PretrainedConfig
    from models.vilmodel import GlocalTextPathNavCMT
    
    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path)
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                k = k[7:]    
            if '_head' in k or 'sap_fuse' in k:
                new_ckpt_weights['bert.' + k] = v
            else:
                new_ckpt_weights[k] = v
            
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    if args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2
    
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_pano_layers = args.num_pano_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.graph_sprels = args.graph_sprels
    vis_config.glocal_fuse = args.fusion == 'dynamic'

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_pano_embedding = args.fix_pano_embedding
    vis_config.fix_local_branch = args.fix_local_branch

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = False
        
    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
    
    # # Apply LoRA if enabled
    # if hasattr(args, 'use_lora') and args.use_lora:
    #     # Configure LoRA
    #     lora_config = LoraConfig(
    #         r=args.lora_r,                      # dimension of the low-rank matrices
    #         lora_alpha=args.lora_alpha,         # scaling factor
    #         target_modules=args.lora_target_modules,  # which modules to apply LoRA to
    #         lora_dropout=args.lora_dropout,     # dropout probability for LoRA layers
    #         bias="none",                        # don't train bias parameters
    #         task_type="SEQ_CLS"                 # task type for the model
    #     )
        
    #     # Convert the model to LoRA
    #     visual_model = get_peft_model(visual_model, lora_config)
        
    #     # Print trainable parameters
    #     visual_model.print_trainable_parameters()
        
    return visual_model
