use half::f16;
use ndarray::{s, Array2, Array4, ArrayView, Ix3};
use once_cell::sync::OnceCell;
use ort::{
    execution_providers::ExecutionProvider,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::Value,
};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

use crate::inference::error::InferenceError;

const MAX_LENGTH: usize = 4096; // Max length of the generated text
const VOCAB_SIZE: usize = 200064; // Phi-4-mini-instruct vocabulary size

static PHI4MINI_INSTANCE: OnceCell<Mutex<Phi4MiniInference>> = OnceCell::new();
static TOKENIZER_INSTANCE: OnceCell<Tokenizer> = OnceCell::new();

pub struct Phi4MiniInference {
    session: Session,
    eos_token_id: u32,
    end_token_id: u32,
    use_cuda: bool,
}

impl Phi4MiniInference {
    pub fn new(model_dir: &Path) -> Result<Self, InferenceError> {
        let use_cuda = ort::execution_providers::CUDAExecutionProvider::default()
            .is_available()
            .unwrap_or(false);

        let model_variant = if use_cuda { "gpu" } else { "cpu" };
        let model_path =
            model_dir.join(format!("phi-4-mini-instruct/{}/model.onnx", model_variant));
        let model_data_path = model_dir.join(format!(
            "phi-4-mini-instruct/{}/model.onnx.data",
            model_variant
        ));

        if !model_path.exists() {
            return Err(InferenceError::PreprocessingError {
                operation: "verify model files".to_string(),
                message: format!("Model file not found: {}", model_path.display()),
            });
        }
        if !model_data_path.exists() {
            return Err(InferenceError::PreprocessingError {
                operation: "verify model files".to_string(),
                message: format!("Model data file not found: {}", model_data_path.display()),
            });
        }

        let session = Session::builder()
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: model_path.clone(),
                source,
            })?
            .with_execution_providers([
                ort::execution_providers::TensorRTExecutionProvider::default()
                    .with_device_id(0)
                    .with_engine_cache(true)
                    .with_engine_cache_path("/workspaces/DocYouMeant/models/trt_engines")
                    .with_engine_cache_prefix("docyoumeant_")
                    .with_max_workspace_size(5 << 30)
                    .with_fp16(true)
                    .with_timing_cache(true)
                    .build(),
            ])?
            .commit_from_file(&model_path)
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: model_path,
                source,
            })?;

        let tokenizer_path = Path::new("models/tokenizer/phi-4-mini-instruct/tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            InferenceError::PreprocessingError {
                operation: "load tokenizer for EOS token".to_string(),
                message: format!("Failed to load tokenizer: {e}"),
            }
        })?;

        let eos_encoding = tokenizer.encode("<|endoftext|>", false).map_err(|e| {
            InferenceError::PreprocessingError {
                operation: "encode EOS token".to_string(),
                message: format!("Failed to encode EOS token: {e}"),
            }
        })?;

        let eos_token_id = eos_encoding.get_ids().first().copied().ok_or_else(|| {
            InferenceError::PreprocessingError {
                operation: "get EOS token ID".to_string(),
                message: "Failed to get EOS token ID".to_string(),
            }
        })?;

        let end_encoding =
            tokenizer
                .encode("<|end|>", false)
                .map_err(|e| InferenceError::PreprocessingError {
                    operation: "encode end token".to_string(),
                    message: format!("Failed to encode end token: {e}"),
                })?;

        let end_token_id = end_encoding.get_ids().first().copied().ok_or_else(|| {
            InferenceError::PreprocessingError {
                operation: "get end token ID".to_string(),
                message: "Failed to get end token ID".to_string(),
            }
        })?;

        Ok(Self {
            session,
            eos_token_id,
            end_token_id,
            use_cuda,
        })
    }

    pub fn get_or_init() -> Result<(), InferenceError> {
        PHI4MINI_INSTANCE.get_or_try_init(|| {
            let model_dir = Path::new("models/onnx");
            Self::new(model_dir).map(Mutex::new)
        })?;
        Self::get_tokenizer()?;
        Ok(())
    }

    pub fn get_tokenizer() -> Result<&'static Tokenizer, InferenceError> {
        TOKENIZER_INSTANCE.get_or_try_init(|| {
            let tokenizer_path = "models/tokenizer/phi-4-mini-instruct/tokenizer.json";
            Tokenizer::from_file(tokenizer_path).map_err(|e| InferenceError::PreprocessingError {
                operation: "load tokenizer".to_string(),
                message: format!("Failed to load tokenizer: {e}"),
            })
        })
    }

    pub fn with_instance<F, R>(f: F) -> Result<R, InferenceError>
    where
        F: FnOnce(&mut Phi4MiniInference) -> Result<R, InferenceError>,
    {
        let instance = PHI4MINI_INSTANCE.get_or_try_init(|| {
            let model_dir = Path::new("models/onnx");
            Self::new(model_dir).map(Mutex::new)
        })?;

        let mut model = instance
            .lock()
            .map_err(|e| InferenceError::ProcessingError {
                message: format!("Failed to lock Phi4MiniInference instance: {e}"),
            })?;

        f(&mut model)
    }

    fn format_chat_template(&self, user_message: &str, system_message: Option<&str>) -> String {
        let system = system_message.unwrap_or("You are a helpful AI assistant.");
        format!("<|system|>{system}<|end|><|user|>{user_message}<|end|><|assistant|>")
    }

    fn detect_token_loop(tokens: &[i64]) -> Option<usize> {
        let len = tokens.len();

        if len < 60 {
            return None;
        }

        for k in 1..=3 {
            let required_reps = 10;
            if len >= required_reps * k {
                let mut all_match = true;
                let last_slice = &tokens[len - k..];
                for i in 1..required_reps {
                    let prev_slice = &tokens[len - (i + 1) * k..len - i * k];
                    if last_slice != prev_slice {
                        all_match = false;
                        break;
                    }
                }
                if all_match {
                    return Some(k);
                }
            }
        }

        for k in 10..=100.min(len / 4) {
            let required_reps = 4;
            if len >= required_reps * k {
                let mut all_match = true;
                let last_slice = &tokens[len - k..];
                for i in 1..required_reps {
                    let prev_slice = &tokens[len - (i + 1) * k..len - i * k];
                    if last_slice != prev_slice {
                        all_match = false;
                        break;
                    }
                }
                if all_match {
                    return Some(k);
                }
            }
        }

        None
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        tokenizer: &Tokenizer,
        system_message: Option<&str>,
    ) -> Result<String, InferenceError> {
        let formatted_prompt = self.format_chat_template(prompt, system_message);

        let encoding = tokenizer.encode(formatted_prompt, true).map_err(|e| {
            InferenceError::PreprocessingError {
                operation: "tokenize input".to_string(),
                message: format!("Error encoding: {e:?}"),
            }
        })?;

        let input_ids_vec: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut input_ids: Array2<i64> =
            Array2::from_shape_vec((1, input_ids_vec.len()), input_ids_vec).map_err(|e| {
                InferenceError::PreprocessingError {
                    operation: "create input_ids array".to_string(),
                    message: format!("Shape error: {e}"),
                }
            })?;

        let attention_mask_vec: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| mask as i64)
            .collect();
        let mut attention_mask: Array2<i64> =
            Array2::from_shape_vec((1, attention_mask_vec.len()), attention_mask_vec).map_err(
                |e| InferenceError::PreprocessingError {
                    operation: "create attention_mask array".to_string(),
                    message: format!("Shape error: {e}"),
                },
            )?;

        let mut past_key_values: Vec<Value> = Vec::with_capacity(64);
        for _ in 0..32 {
            if self.use_cuda {
                let empty = Array4::from_elem((1, 8, 0, 128), f16::from_f32(0.0));
                past_key_values.push(
                    Value::from_array(empty)
                        .map_err(|e| InferenceError::PreprocessingError {
                            operation: "create empty kv tensor".to_string(),
                            message: e.to_string(),
                        })?
                        .into(),
                );
                let empty = Array4::from_elem((1, 8, 0, 128), f16::from_f32(0.0));
                past_key_values.push(
                    Value::from_array(empty)
                        .map_err(|e| InferenceError::PreprocessingError {
                            operation: "create empty kv tensor".to_string(),
                            message: e.to_string(),
                        })?
                        .into(),
                );
            } else {
                let empty = Array4::<f32>::zeros((1, 8, 0, 128));
                past_key_values.push(
                    Value::from_array(empty)
                        .map_err(|e| InferenceError::PreprocessingError {
                            operation: "create empty kv tensor".to_string(),
                            message: e.to_string(),
                        })?
                        .into(),
                );
                let empty = Array4::<f32>::zeros((1, 8, 0, 128));
                past_key_values.push(
                    Value::from_array(empty)
                        .map_err(|e| InferenceError::PreprocessingError {
                            operation: "create empty kv tensor".to_string(),
                            message: e.to_string(),
                        })?
                        .into(),
                );
            }
        }

        let mut generated_tokens: Vec<i64> = Vec::new();
        let mut current_position = input_ids.shape()[1] as i64;

        let output_mem_info = if self.use_cuda {
            MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(|e| InferenceError::PreprocessingError {
                operation: "create cuda memory info".to_string(),
                message: e.to_string(),
            })?
        } else {
            MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Arena,
                MemoryType::Default,
            )
            .map_err(|e| InferenceError::PreprocessingError {
                operation: "create cpu memory info".to_string(),
                message: e.to_string(),
            })?
        };

        let cpu_mem_info = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Arena,
            MemoryType::Default,
        )
        .map_err(|e| InferenceError::PreprocessingError {
            operation: "create cpu memory info".to_string(),
            message: e.to_string(),
        })?;

        for _ in 0..MAX_LENGTH {
            let mut binding =
                self.session
                    .create_binding()
                    .map_err(|e| InferenceError::ModelExecutionError {
                        operation: "create binding".to_string(),
                        source: e,
                    })?;

            let attention_mask_len = attention_mask.shape()[1];

            if self.use_cuda {
                let position_ids = Array2::from_shape_fn((1, input_ids.shape()[1]), |(_, j)| {
                    current_position - input_ids.shape()[1] as i64 + j as i64
                });
                let position_ids_val: Value = Value::from_array(position_ids)
                    .map_err(|e| InferenceError::PreprocessingError {
                        operation: "bind position_ids".to_string(),
                        message: e.to_string(),
                    })?
                    .into();
                binding
                    .bind_input("position_ids", &position_ids_val)
                    .map_err(|e| InferenceError::ModelExecutionError {
                        operation: "bind position_ids".to_string(),
                        source: e,
                    })?;
            }

            let input_ids_val: Value = Value::from_array(input_ids)
                .map_err(|e| InferenceError::PreprocessingError {
                    operation: "bind input_ids".to_string(),
                    message: e.to_string(),
                })?
                .into();
            binding
                .bind_input("input_ids", &input_ids_val)
                .map_err(|e| InferenceError::ModelExecutionError {
                    operation: "bind input_ids".to_string(),
                    source: e,
                })?;

            let attention_mask_val: Value = Value::from_array(attention_mask)
                .map_err(|e| InferenceError::PreprocessingError {
                    operation: "bind attention_mask".to_string(),
                    message: e.to_string(),
                })?
                .into();
            binding
                .bind_input("attention_mask", &attention_mask_val)
                .map_err(|e| InferenceError::ModelExecutionError {
                    operation: "bind attention_mask".to_string(),
                    source: e,
                })?;

            for i in 0..32 {
                binding
                    .bind_input(
                        format!("past_key_values.{}.key", i),
                        &past_key_values[i * 2],
                    )
                    .map_err(|e| InferenceError::ModelExecutionError {
                        operation: format!("bind past_key_values.{}.key", i),
                        source: e,
                    })?;
                binding
                    .bind_input(
                        format!("past_key_values.{}.value", i),
                        &past_key_values[i * 2 + 1],
                    )
                    .map_err(|e| InferenceError::ModelExecutionError {
                        operation: format!("bind past_key_values.{}.value", i),
                        source: e,
                    })?;
            }

            binding
                .bind_output_to_device("logits", &cpu_mem_info)
                .map_err(|e| InferenceError::ModelExecutionError {
                    operation: "bind logits".to_string(),
                    source: e,
                })?;

            for i in 0..32 {
                binding
                    .bind_output_to_device(format!("present.{}.key", i), &output_mem_info)
                    .map_err(|e| InferenceError::ModelExecutionError {
                        operation: format!("bind present.{}.key", i),
                        source: e,
                    })?;
                binding
                    .bind_output_to_device(format!("present.{}.value", i), &output_mem_info)
                    .map_err(|e| InferenceError::ModelExecutionError {
                        operation: format!("bind present.{}.value", i),
                        source: e,
                    })?;
            }

            let mut outputs = self.session.run_binding(&binding).map_err(|e| {
                InferenceError::ModelExecutionError {
                    operation: "run binding".to_string(),
                    source: e,
                }
            })?;

            let logits_value = outputs.remove("logits").unwrap();

            let token_id = if self.use_cuda {
                let logits: ArrayView<f16, _> = logits_value
                    .try_extract_array::<f16>()
                    .map_err(|source| InferenceError::ModelExecutionError {
                        operation: "extract logits".to_string(),
                        source,
                    })?
                    .into_dimensionality::<Ix3>()
                    .map_err(|e| InferenceError::PredictionError {
                        operation: "reshape logits".to_string(),
                        message: format!("Failed to reshape logits: {e}"),
                    })?;

                logits
                    .slice(s![0, -1, ..VOCAB_SIZE])
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0 as i64
            } else {
                let logits: ArrayView<f32, _> = logits_value
                    .try_extract_array::<f32>()
                    .map_err(|source| InferenceError::ModelExecutionError {
                        operation: "extract logits".to_string(),
                        source,
                    })?
                    .into_dimensionality::<Ix3>()
                    .map_err(|e| InferenceError::PredictionError {
                        operation: "reshape logits".to_string(),
                        message: format!("Failed to reshape logits: {e}"),
                    })?;

                logits
                    .slice(s![0, -1, ..VOCAB_SIZE])
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0 as i64
            };

            if token_id == self.eos_token_id as i64 || token_id == self.end_token_id as i64 {
                break;
            }

            generated_tokens.push(token_id);

            if generated_tokens.len().is_multiple_of(10) {
                if let Some(pattern_len) = Self::detect_token_loop(&generated_tokens) {
                    let len = generated_tokens.len();
                    if pattern_len <= 3 {
                        generated_tokens.truncate(len - pattern_len * 9);
                    } else {
                        generated_tokens.truncate(len - pattern_len * 3);
                    }
                    break;
                }
            }

            let mut new_past_key_values = Vec::with_capacity(64);
            for i in 0..32 {
                new_past_key_values.push(outputs.remove(format!("present.{}.key", i)).unwrap());
                new_past_key_values.push(outputs.remove(format!("present.{}.value", i)).unwrap());
            }
            past_key_values = new_past_key_values;

            input_ids = Array2::from_elem((1, 1), token_id);
            attention_mask = Array2::ones((1, attention_mask_len + 1));
            current_position += 1;
        }

        let output_ids: Vec<u32> = generated_tokens.iter().map(|&id| id as u32).collect();
        let generated_text =
            tokenizer
                .decode(&output_ids, false)
                .map_err(|e| InferenceError::PredictionError {
                    operation: "decode tokens".to_string(),
                    message: format!("Error decoding: {e:?}"),
                })?;

        Ok(generated_text)
    }
}
