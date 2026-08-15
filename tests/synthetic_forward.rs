use std::{collections::HashMap, fs};

use candle_core::{Device, Tensor, safetensors};
use qwen_engine::{Checkpoint, qwen::Model};
use serde_json::json;

fn insert(tensors: &mut HashMap<String, Tensor>, name: String, shape: &[usize]) {
    let seed = name.bytes().fold(0usize, |sum, byte| sum + byte as usize);
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count)
        .map(|i| (((i + seed) % 17) as f32 - 8.) * 0.01)
        .collect();
    tensors.insert(name, Tensor::from_vec(values, shape, &Device::Cpu).unwrap());
}

#[test]
fn complete_hybrid_model_prefills_and_decodes() {
    let dir = tempfile::tempdir().unwrap();
    let config = json!({
        "model_type": "qwen3_next", "vocab_size": 8, "hidden_size": 4,
        "intermediate_size": 6, "num_hidden_layers": 4,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
        "linear_conv_kernel_dim": 2, "linear_key_head_dim": 2,
        "linear_value_head_dim": 2, "linear_num_key_heads": 1,
        "linear_num_value_heads": 1, "moe_intermediate_size": 2,
        "shared_expert_intermediate_size": 2, "num_experts_per_tok": 1,
        "num_experts": 2, "decoder_sparse_step": 1, "mlp_only_layers": [],
        "full_attention_interval": 4, "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "partial_rotary_factor": 1.0,
        "norm_topk_prob": true, "tie_word_embeddings": false,
        "attention_bias": false, "max_position_embeddings": 32,
        "bos_token_id": 1, "eos_token_id": 2
    });
    fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut tensors = HashMap::new();
    insert(&mut tensors, "model.embed_tokens.weight".into(), &[8, 4]);
    insert(&mut tensors, "model.norm.weight".into(), &[4]);
    insert(&mut tensors, "lm_head.weight".into(), &[8, 4]);
    for layer in 0..4 {
        let p = format!("model.layers.{layer}");
        insert(&mut tensors, format!("{p}.input_layernorm.weight"), &[4]);
        insert(
            &mut tensors,
            format!("{p}.post_attention_layernorm.weight"),
            &[4],
        );
        if layer == 3 {
            let a = format!("{p}.self_attn");
            insert(&mut tensors, format!("{a}.q_proj.weight"), &[8, 4]);
            insert(&mut tensors, format!("{a}.k_proj.weight"), &[2, 4]);
            insert(&mut tensors, format!("{a}.v_proj.weight"), &[2, 4]);
            insert(&mut tensors, format!("{a}.o_proj.weight"), &[4, 4]);
            insert(&mut tensors, format!("{a}.q_norm.weight"), &[2]);
            insert(&mut tensors, format!("{a}.k_norm.weight"), &[2]);
        } else {
            let a = format!("{p}.linear_attn");
            insert(&mut tensors, format!("{a}.in_proj_qkvz.weight"), &[8, 4]);
            insert(&mut tensors, format!("{a}.in_proj_ba.weight"), &[2, 4]);
            insert(&mut tensors, format!("{a}.conv1d.weight"), &[6, 1, 2]);
            insert(&mut tensors, format!("{a}.dt_bias"), &[1]);
            insert(&mut tensors, format!("{a}.A_log"), &[1]);
            insert(&mut tensors, format!("{a}.norm.weight"), &[2]);
            insert(&mut tensors, format!("{a}.out_proj.weight"), &[4, 2]);
        }
        let m = format!("{p}.mlp");
        insert(&mut tensors, format!("{m}.gate.weight"), &[2, 4]);
        for expert in 0..2 {
            let e = format!("{m}.experts.{expert}");
            insert(&mut tensors, format!("{e}.gate_proj.weight"), &[2, 4]);
            insert(&mut tensors, format!("{e}.up_proj.weight"), &[2, 4]);
            insert(&mut tensors, format!("{e}.down_proj.weight"), &[4, 2]);
        }
        insert(
            &mut tensors,
            format!("{m}.shared_expert.gate_proj.weight"),
            &[2, 4],
        );
        insert(
            &mut tensors,
            format!("{m}.shared_expert.up_proj.weight"),
            &[2, 4],
        );
        insert(
            &mut tensors,
            format!("{m}.shared_expert.down_proj.weight"),
            &[4, 2],
        );
        insert(
            &mut tensors,
            format!("{m}.shared_expert_gate.weight"),
            &[1, 4],
        );
    }
    safetensors::save(&tensors, dir.path().join("model.safetensors")).unwrap();

    let model = Model::new(Checkpoint::open(dir.path()).unwrap());
    let mut state = model.new_state();
    let (prefill, _) = model.forward(&[1, 3], &mut state, None).unwrap();
    assert_eq!(prefill.dims(), &[2, 8]);
    assert!(
        prefill
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .flatten()
            .all(|v| v.is_finite())
    );
    let (decode, _) = model.forward(&[4], &mut state, None).unwrap();
    assert_eq!(decode.dims(), &[1, 8]);
    assert_eq!(state.position, 3);

    let mut whole_state = model.new_state();
    let (whole, _) = model.forward(&[1, 3, 4], &mut whole_state, None).unwrap();
    let whole_last = whole.narrow(0, 2, 1).unwrap().to_vec2::<f32>().unwrap();
    let mut step_state = model.new_state();
    model.forward(&[1], &mut step_state, None).unwrap();
    model.forward(&[3], &mut step_state, None).unwrap();
    let (step, _) = model.forward(&[4], &mut step_state, None).unwrap();
    let step_last = step.to_vec2::<f32>().unwrap();
    for (prefill, incremental) in whole_last[0].iter().zip(&step_last[0]) {
        assert!(
            (prefill - incremental).abs() < 1e-5,
            "{prefill} != {incremental}"
        );
    }
}
