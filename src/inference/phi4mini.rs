//! Phi-4-mini-instruct language model for text generation.
//!
//! This module provides text generation capabilities using Microsoft's Phi-4-mini-instruct
//! model, a compact but powerful language model optimized for instruction following.
//! The model can be used for various document understanding tasks such as summarization,
//! question answering, and content extraction.

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
use crate::utils::config::AppConfig;

/// Maximum number of tokens to generate in a single response.
const MAX_LENGTH: usize = 4096;

/// Size of the Phi-4-mini-instruct vocabulary.
const VOCAB_SIZE: usize = 200_064;

/// Singleton instance for the model.
static PHI4MINI_INSTANCE: OnceCell<Mutex<Phi4MiniInference>> = OnceCell::new();
/// Singleton instance for the tokenizer.
static TOKENIZER_INSTANCE: OnceCell<Tokenizer> = OnceCell::new();

/// Phi-4-mini-instruct language model for text generation.
///
/// `Phi4MiniInference` wraps an ONNX Runtime session configured for autoregressive
/// text generation with KV caching for efficient incremental decoding.
///
/// # Fields
///
/// - `session`: ONNX Runtime session with the loaded model
/// - `eos_token_id`: Token ID for end-of-sequence (`<|endoftext|>`)
/// - `end_token_id`: Token ID for assistant turn end (`<|end|>`)
/// - `use_cuda`: Whether CUDA acceleration is available and enabled
pub struct Phi4MiniInference {
    session: Session,
    eos_token_id: u32,
    end_token_id: u32,
    use_cuda: bool,
}

impl Phi4MiniInference {
    /// Creates a new Phi-4-mini-instruct inference instance.
    ///
    /// Loads the ONNX model and tokenizer, automatically selecting between
    /// GPU (CUDA) and CPU variants based on hardware availability.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to the models directory containing `phi-4-mini-instruct/`
    ///
    /// # Returns
    ///
    /// * `Ok(Phi4MiniInference)` - Initialized model ready for generation
    /// * `Err(InferenceError)` - If model files are missing or cannot be loaded
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
            .commit_from_file(&model_path)
            .map_err(|source| InferenceError::ModelFileLoadError {
                path: model_path,
                source,
            })?;

        let config = AppConfig::get();
        let tokenizer_path = config.model_path("tokenizer/phi-4-mini-instruct/tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
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

    /// Pre-initializes the model and tokenizer singletons.
    ///
    /// Call this method during application startup to eagerly load the model.
    /// This can take several seconds depending on hardware and model size.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Model and tokenizer successfully initialized
    /// * `Err(InferenceError)` - If initialization fails
    pub fn get_or_init() -> Result<(), InferenceError> {
        PHI4MINI_INSTANCE.get_or_try_init(|| {
            let config = AppConfig::get();
            let model_path = config.model_path("onnx");
            let model_dir = Path::new(&model_path);
            Self::new(model_dir).map(Mutex::new)
        })?;
        Self::get_tokenizer()?;
        Ok(())
    }

    /// Returns a reference to the shared tokenizer instance.
    ///
    /// The tokenizer is loaded lazily on first access and cached for
    /// subsequent calls.
    ///
    /// # Returns
    ///
    /// * `Ok(&'static Tokenizer)` - Reference to the shared tokenizer
    /// * `Err(InferenceError)` - If the tokenizer file cannot be loaded
    pub fn get_tokenizer() -> Result<&'static Tokenizer, InferenceError> {
        TOKENIZER_INSTANCE.get_or_try_init(|| {
            let config = AppConfig::get();
            let tokenizer_path = config.model_path("tokenizer/phi-4-mini-instruct/tokenizer.json");
            Tokenizer::from_file(&tokenizer_path).map_err(|e| InferenceError::PreprocessingError {
                operation: "load tokenizer".to_string(),
                message: format!("Failed to load tokenizer: {e}"),
            })
        })
    }

    /// Executes a function with access to the model instance.
    ///
    /// This is the primary way to interact with the model. It handles singleton
    /// initialization and mutex locking automatically.
    ///
    /// # Type Parameters
    ///
    /// * `F` - Function type that takes `&mut Phi4MiniInference` and returns `Result<R, InferenceError>`
    /// * `R` - Return type of the function
    ///
    /// # Arguments
    ///
    /// * `f` - Function to execute with the model instance
    ///
    /// # Returns
    ///
    /// The result of the provided function.
    pub fn with_instance<F, R>(f: F) -> Result<R, InferenceError>
    where
        F: FnOnce(&mut Phi4MiniInference) -> Result<R, InferenceError>,
    {
        let instance = PHI4MINI_INSTANCE.get_or_try_init(|| {
            let config = AppConfig::get();
            let model_path = config.model_path("onnx");
            let model_dir = Path::new(&model_path);
            Self::new(model_dir).map(Mutex::new)
        })?;

        let mut model = instance
            .lock()
            .map_err(|e| InferenceError::ProcessingError {
                message: format!("Failed to lock Phi4MiniInference instance: {e}"),
            })?;

        f(&mut model)
    }

    /// Formats a prompt using the Phi-4 chat template.
    ///
    /// Wraps the user message in the expected chat format with system and
    /// user turn markers.
    ///
    /// # Arguments
    ///
    /// * `user_message` - The user's input prompt
    /// * `system_message` - Optional system prompt (defaults to generic assistant)
    ///
    /// # Returns
    ///
    /// Formatted string: `<|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>`
    fn format_chat_template(&self, user_message: &str, system_message: Option<&str>) -> String {
        let system = system_message.unwrap_or("You are a helpful AI assistant.");
        format!("<|system|>{system}<|end|><|user|>{user_message}<|end|><|assistant|>")
    }

    /// Prepares tokenized inputs for model inference.
    ///
    /// Formats the prompt using the chat template and tokenizes it to create
    /// the input tensors required by the model.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Raw user prompt text
    /// * `tokenizer` - Tokenizer instance for encoding
    /// * `system_message` - Optional system prompt
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `Array2<i64>` - Input IDs tensor of shape (1, seq_len)
    /// - `Array2<i64>` - Attention mask tensor of shape (1, seq_len)
    fn prepare_inputs(
        &self,
        prompt: &str,
        tokenizer: &Tokenizer,
        system_message: Option<&str>,
    ) -> Result<(Array2<i64>, Array2<i64>), InferenceError> {
        let formatted_prompt = self.format_chat_template(prompt, system_message);

        let encoding = tokenizer.encode(formatted_prompt, true).map_err(|e| {
            InferenceError::PreprocessingError {
                operation: "tokenize input".to_string(),
                message: format!("Error encoding: {e:?}"),
            }
        })?;

        let input_ids_vec: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let input_ids: Array2<i64> =
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
        let attention_mask: Array2<i64> =
            Array2::from_shape_vec((1, attention_mask_vec.len()), attention_mask_vec).map_err(
                |e| InferenceError::PreprocessingError {
                    operation: "create attention_mask array".to_string(),
                    message: format!("Shape error: {e}"),
                },
            )?;

        Ok((input_ids, attention_mask))
    }

    /// Initializes empty KV cache tensors for all 32 transformer layers.
    ///
    /// Creates 64 tensors (key and value for each layer) with zero sequence length,
    /// ready to accumulate attention states during generation.
    ///
    /// # Returns
    ///
    /// Vector of 64 ONNX values representing empty KV caches.
    /// Shape per tensor: (1, 8, 0, 128) where 8 is the number of attention heads.
    fn init_past_key_values(&self) -> Result<Vec<Value>, InferenceError> {
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
        Ok(past_key_values)
    }

    /// Creates memory allocation information for ONNX Runtime.
    ///
    /// Configures where input/output tensors should be allocated based on
    /// the available hardware.
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - Output memory info (CUDA or CPU depending on hardware)
    /// - CPU memory info (always CPU, used for reading logits)
    fn create_memory_info(&self) -> Result<(MemoryInfo, MemoryInfo), InferenceError> {
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

        Ok((output_mem_info, cpu_mem_info))
    }

    /// Extracts the next token from model logits using greedy decoding.
    ///
    /// Takes the argmax over the vocabulary dimension of the last position's
    /// logits to determine the most likely next token.
    ///
    /// # Arguments
    ///
    /// * `logits_value` - Output logits tensor from the model
    ///
    /// # Returns
    ///
    /// * `Ok(i64)` - Token ID of the most likely next token
    /// * `Err(InferenceError)` - If logits extraction fails
    fn extract_next_token(&self, logits_value: Value) -> Result<i64, InferenceError> {
        if self.use_cuda {
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

            let (token_idx, _) = logits
                .slice(s![0, -1, ..VOCAB_SIZE])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| InferenceError::PredictionError {
                    operation: "find max logit".to_string(),
                    message: "Empty logits tensor".to_string(),
                })?;
            Ok(token_idx as i64)
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

            let (token_idx, _) = logits
                .slice(s![0, -1, ..VOCAB_SIZE])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| InferenceError::PredictionError {
                    operation: "find max logit".to_string(),
                    message: "Empty logits tensor".to_string(),
                })?;
            Ok(token_idx as i64)
        }
    }

    /// Checks if a token is a stop token that should end generation.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Token ID to check
    ///
    /// # Returns
    ///
    /// `true` if the token is either `<|endoftext|>` or `<|end|>`.
    fn is_stop_token(&self, token_id: i64) -> bool {
        token_id == self.eos_token_id as i64 || token_id == self.end_token_id as i64
    }

    /// Decodes a sequence of token IDs back to text.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Slice of generated token IDs
    /// * `tokenizer` - Tokenizer for decoding
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Decoded text
    /// * `Err(InferenceError)` - If decoding fails
    fn decode_tokens(
        &self,
        tokens: &[i64],
        tokenizer: &Tokenizer,
    ) -> Result<String, InferenceError> {
        let output_ids: Vec<u32> = tokens.iter().map(|&id| id as u32).collect();
        tokenizer
            .decode(&output_ids, false)
            .map_err(|e| InferenceError::PredictionError {
                operation: "decode tokens".to_string(),
                message: format!("Error decoding: {e:?}"),
            })
    }

    /// Generates text completion for a given prompt.
    ///
    /// Performs autoregressive text generation using the Phi-4-mini-instruct model.
    /// Generation continues until a stop token is produced or the maximum length
    /// is reached.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user's input prompt
    /// * `tokenizer` - Tokenizer for encoding/decoding
    /// * `system_message` - Optional system prompt to guide model behavior
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Generated text response
    /// * `Err(InferenceError)` - If generation fails
    pub fn generate(
        &mut self,
        prompt: &str,
        tokenizer: &Tokenizer,
        system_message: Option<&str>,
    ) -> Result<String, InferenceError> {
        let (mut input_ids, mut attention_mask) =
            self.prepare_inputs(prompt, tokenizer, system_message)?;
        let mut past_key_values = self.init_past_key_values()?;
        let (output_mem_info, cpu_mem_info) = self.create_memory_info()?;

        let mut generated_tokens: Vec<i64> = Vec::new();
        let mut current_position = input_ids.shape()[1] as i64;

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

            let logits_value =
                outputs
                    .remove("logits")
                    .ok_or_else(|| InferenceError::ProcessingError {
                        message: "Missing output 'logits'".to_string(),
                    })?;

            let mut new_past_key_values = Vec::with_capacity(64);
            for i in 0..32 {
                let key_name = format!("present.{}.key", i);
                let value_name = format!("present.{}.value", i);

                new_past_key_values.push(outputs.remove(&key_name).ok_or_else(|| {
                    InferenceError::ProcessingError {
                        message: format!("Missing output '{}'", key_name),
                    }
                })?);
                new_past_key_values.push(outputs.remove(&value_name).ok_or_else(|| {
                    InferenceError::ProcessingError {
                        message: format!("Missing output '{}'", value_name),
                    }
                })?);
            }

            drop(outputs);

            let token_id = self.extract_next_token(logits_value)?;

            if self.is_stop_token(token_id) {
                break;
            }

            generated_tokens.push(token_id);
            past_key_values = new_past_key_values;

            input_ids = Array2::from_elem((1, 1), token_id);
            attention_mask = Array2::ones((1, attention_mask_len + 1));
            current_position += 1;
        }

        self.decode_tokens(&generated_tokens, tokenizer)
    }
}
