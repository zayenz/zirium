//! Explicit, static dialect registration descriptors.

use std::sync::OnceLock;

use crate::{
    SyntaxKind,
    parser::DialectParser,
    semantic::{Document, OperationId, RegisteredLowering, RegisteredLoweringContext},
};

pub type OperationParser = fn(&mut DialectParser<'_, '_>) -> Result<(), crate::CompactError>;
pub type OperationLowerer = fn(&RegisteredLoweringContext<'_>) -> Option<RegisteredLowering>;
pub type OperationVerifier = fn(&Document, OperationId) -> Result<(), &'static str>;
pub type OperationPrinter = fn(&Document, OperationId) -> Option<String>;
pub type ValueParser = fn(&str) -> bool;
pub type ValueLowerer = fn(&str) -> Option<String>;
pub type ValueVerifier = fn(&str) -> Result<(), &'static str>;
pub type ValuePrinter = fn(&str) -> Option<String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    Ssacfg,
    Graph,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SymbolDescriptor {
    pub defines_symbol: bool,
    pub symbol_table: bool,
    pub uses_symbols: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionDescriptor {
    pub kind: RegionKind,
    pub isolated_from_above: bool,
}

impl Default for RegionDescriptor {
    fn default() -> Self {
        Self {
            kind: RegionKind::Ssacfg,
            isolated_from_above: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperationSchema {
    pub operands: OperandCount,
    pub results: ResultCount,
    pub required_attributes: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResultCount {
    Exact(usize),
    Variadic,
}

impl ResultCount {
    pub(crate) fn accepts(self, count: usize) -> bool {
        matches!(self, Self::Variadic) || matches!(self, Self::Exact(expected) if expected == count)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandCount {
    Exact(usize),
    Variadic,
}

impl OperandCount {
    pub(crate) fn accepts(self, count: usize) -> bool {
        matches!(self, Self::Variadic) || matches!(self, Self::Exact(expected) if expected == count)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssemblyProgram {
    Module,
    Function,
    Call,
    ConditionalBranch,
    TypedAttribute,
    BinaryOperands,
    OptionalTypedOperands,
    TypedSuccessor,
}

impl AssemblyProgram {
    const fn operation_name(self) -> &'static str {
        match self {
            Self::Module => "builtin.module",
            Self::Function => "func.func",
            Self::Call => "func.call",
            Self::ConditionalBranch => "cf.cond_br",
            Self::TypedAttribute => "arith.constant",
            Self::BinaryOperands => "arith.addi",
            Self::OptionalTypedOperands => "func.return",
            Self::TypedSuccessor => "cf.br",
        }
    }

    pub(crate) const fn inherent_attribute(self) -> Option<&'static str> {
        match self {
            Self::Module => Some("sym_name"),
            Self::Function => Some("sym_name"),
            Self::Call => Some("callee"),
            Self::ConditionalBranch => None,
            Self::TypedAttribute => Some("value"),
            Self::BinaryOperands => Some("overflowFlags"),
            Self::OptionalTypedOperands | Self::TypedSuccessor => None,
        }
    }

    pub(crate) fn lower(
        self,
        context: &RegisteredLoweringContext<'_>,
    ) -> Option<RegisteredLowering> {
        match self {
            Self::Module => lower_module(context),
            Self::Function => lower_function(context),
            Self::Call => lower_call(context),
            Self::ConditionalBranch => lower_no_results("cf.cond_br"),
            Self::TypedAttribute => lower_arith_constant(context),
            Self::BinaryOperands => lower_addi(context),
            Self::OptionalTypedOperands => lower_return(context),
            Self::TypedSuccessor => lower_no_results(self.operation_name()),
        }
    }

    pub(crate) fn verify(
        self,
        document: &Document,
        operation: OperationId,
    ) -> Result<(), &'static str> {
        match self {
            Self::Module => crate::semantic::verify_builtin_module(document, operation),
            Self::Function => crate::semantic::verify_func_func(document, operation),
            Self::Call => crate::semantic::verify_func_call(document, operation),
            Self::ConditionalBranch => crate::semantic::verify_cf_cond_br(document, operation),
            Self::TypedAttribute => verify_arith_constant(document, operation),
            Self::BinaryOperands => verify_addi(document, operation),
            Self::OptionalTypedOperands => crate::semantic::verify_func_return(document, operation),
            Self::TypedSuccessor => crate::semantic::verify_cf_br(document, operation),
        }
    }

    pub(crate) fn print(self, document: &Document, operation: OperationId) -> Option<String> {
        match self {
            Self::Module => print_module(document, operation),
            Self::Function => print_function(document, operation),
            Self::Call => print_call(document, operation),
            Self::ConditionalBranch => print_cond_branch(document, operation),
            Self::TypedAttribute => print_arith_constant(document, operation),
            Self::BinaryOperands => print_addi(document, operation),
            Self::OptionalTypedOperands => print_return(document, operation),
            Self::TypedSuccessor => print_branch(document, operation),
        }
    }
}

const fn same_name(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

const fn same_names(left: &[&str], right: &[&str]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if !same_name(left[index], right[index]) {
            return false;
        }
        index += 1;
    }
    true
}

#[derive(Clone, Copy)]
pub struct OperationDescriptor {
    pub name: &'static str,
    pub syntax_kind: SyntaxKind,
    pub parse: Option<OperationParser>,
    pub lower: Option<OperationLowerer>,
    pub verify: Option<OperationVerifier>,
    pub print: Option<OperationPrinter>,
    pub assembly: Option<AssemblyProgram>,
    pub schema: OperationSchema,
    pub regions: &'static [RegionDescriptor],
    pub symbols: SymbolDescriptor,
}

pub struct TypeDescriptor {
    pub name: &'static str,
    pub parse: Option<ValueParser>,
    pub lower: Option<ValueLowerer>,
    pub verify: Option<ValueVerifier>,
    pub print: Option<ValuePrinter>,
}

pub struct AttributeDescriptor {
    pub name: &'static str,
    pub parse: Option<ValueParser>,
    pub lower: Option<ValueLowerer>,
    pub verify: Option<ValueVerifier>,
    pub print: Option<ValuePrinter>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DeclarativeRegistryError {
    UnknownOperation(String),
    DuplicateOperation(String),
    EmptyOperation,
    CoreOperation(String),
}

impl std::fmt::Display for DeclarativeRegistryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownOperation(name) => {
                write!(formatter, "unknown declarative operation: {name}")
            }
            Self::DuplicateOperation(name) => {
                write!(formatter, "duplicate declarative operation: {name}")
            }
            Self::EmptyOperation => write!(formatter, "operation name must not be empty"),
            Self::CoreOperation(name) => write!(
                formatter,
                "operation shape conflicts with core operation: {name}"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationShape {
    FuncLike,
    CallLike,
}

impl std::error::Error for DeclarativeRegistryError {}

pub struct DialectRegistry {
    operations: &'static [OperationDescriptor],
    types: &'static [TypeDescriptor],
    attributes: &'static [AttributeDescriptor],
    operation_shapes: Option<Box<[(String, OperationShape)]>>,
    module_alias: bool,
}

impl DialectRegistry {
    pub const EMPTY: Self = Self::new(&[], &[], &[]);

    pub const fn new(
        operations: &'static [OperationDescriptor],
        types: &'static [TypeDescriptor],
        attributes: &'static [AttributeDescriptor],
    ) -> Self {
        let registry = Self {
            operations,
            types,
            attributes,
            operation_shapes: None,
            module_alias: false,
        };
        let mut index = 0;
        while index < operations.len() {
            assert!(
                !operations[index].name.is_empty(),
                "operation name must not be empty"
            );
            assert!(
                operations[index].assembly.is_some() || operations[index].parse.is_some(),
                "registered custom operation needs syntax handling"
            );
            if let Some(program) = operations[index].assembly {
                assert!(
                    same_name(operations[index].name, program.operation_name()),
                    "assembly program is attached to the wrong operation"
                );
                let schema = operations[index].schema;
                match program {
                    AssemblyProgram::Module => {
                        assert!(
                            matches!(schema.operands, OperandCount::Exact(0))
                                && matches!(schema.results, ResultCount::Exact(0))
                                && same_names(schema.required_attributes, &[])
                                && operations[index].regions.len() == 1
                                && matches!(operations[index].regions[0].kind, RegionKind::Ssacfg)
                                && operations[index].regions[0].isolated_from_above
                                && operations[index].symbols.defines_symbol
                                && operations[index].symbols.symbol_table
                                && !operations[index].symbols.uses_symbols,
                            "module schema is inconsistent"
                        );
                    }
                    AssemblyProgram::Function => {
                        assert!(
                            matches!(schema.operands, OperandCount::Exact(0))
                                && matches!(schema.results, ResultCount::Exact(0))
                                && same_names(
                                    schema.required_attributes,
                                    &["sym_name", "function_type"]
                                )
                                && operations[index].regions.len() == 1
                                && matches!(operations[index].regions[0].kind, RegionKind::Ssacfg)
                                && operations[index].regions[0].isolated_from_above
                                && operations[index].symbols.defines_symbol
                                && !operations[index].symbols.symbol_table
                                && !operations[index].symbols.uses_symbols,
                            "function schema is inconsistent"
                        );
                    }
                    AssemblyProgram::Call => {
                        assert!(
                            matches!(schema.operands, OperandCount::Variadic)
                                && matches!(schema.results, ResultCount::Variadic)
                                && same_names(schema.required_attributes, &["callee"])
                                && operations[index].regions.is_empty()
                                && operations[index].symbols.uses_symbols
                                && !operations[index].symbols.defines_symbol
                                && !operations[index].symbols.symbol_table,
                            "call schema is inconsistent"
                        );
                    }
                    AssemblyProgram::ConditionalBranch => {
                        assert!(
                            matches!(schema.operands, OperandCount::Variadic)
                                && matches!(schema.results, ResultCount::Exact(0))
                                && same_names(schema.required_attributes, &[])
                                && operations[index].regions.is_empty()
                                && !operations[index].symbols.defines_symbol
                                && !operations[index].symbols.symbol_table
                                && !operations[index].symbols.uses_symbols,
                            "conditional branch schema is inconsistent"
                        );
                    }
                    AssemblyProgram::TypedAttribute => {
                        assert!(
                            matches!(schema.operands, OperandCount::Exact(0))
                                && matches!(schema.results, ResultCount::Exact(1))
                                && same_names(schema.required_attributes, &["value"])
                                && operations[index].regions.is_empty()
                                && !operations[index].symbols.defines_symbol
                                && !operations[index].symbols.symbol_table
                                && !operations[index].symbols.uses_symbols,
                            "typed attribute schema is inconsistent"
                        );
                    }
                    AssemblyProgram::BinaryOperands => {
                        assert!(
                            matches!(schema.operands, OperandCount::Exact(2))
                                && matches!(schema.results, ResultCount::Exact(1))
                                && same_names(schema.required_attributes, &[])
                                && operations[index].regions.is_empty()
                                && !operations[index].symbols.defines_symbol
                                && !operations[index].symbols.symbol_table
                                && !operations[index].symbols.uses_symbols,
                            "binary operand schema is inconsistent"
                        );
                    }
                    AssemblyProgram::OptionalTypedOperands => {
                        assert!(
                            matches!(schema.operands, OperandCount::Variadic)
                                && matches!(schema.results, ResultCount::Exact(0))
                                && same_names(schema.required_attributes, &[])
                                && operations[index].regions.is_empty()
                                && !operations[index].symbols.defines_symbol
                                && !operations[index].symbols.symbol_table
                                && !operations[index].symbols.uses_symbols,
                            "optional operand schema is inconsistent"
                        );
                    }
                    AssemblyProgram::TypedSuccessor => {
                        assert!(
                            matches!(schema.operands, OperandCount::Exact(0))
                                && matches!(schema.results, ResultCount::Exact(0))
                                && same_names(schema.required_attributes, &[])
                                && operations[index].regions.is_empty()
                                && !operations[index].symbols.defines_symbol
                                && !operations[index].symbols.symbol_table
                                && !operations[index].symbols.uses_symbols,
                            "successor schema is inconsistent"
                        );
                    }
                }
            }
            index += 1;
        }
        registry
    }

    pub fn operation(&self, name: &str) -> Option<&OperationDescriptor> {
        self.operations
            .iter()
            .find(|descriptor| descriptor.name == name)
    }

    pub fn operation_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.operations.iter().map(|descriptor| descriptor.name)
    }

    pub fn operation_shape(&self, name: &str) -> Option<OperationShape> {
        self.operation_shapes
            .as_deref()?
            .iter()
            .find_map(|(candidate, shape)| (candidate == name).then_some(*shape))
    }

    /// Builds an owned registry containing the core operations plus caller-named operations
    /// assigned supported [`OperationShape`] variants.
    pub fn with_operation_shapes(
        operation_shapes: &[(&str, OperationShape)],
    ) -> Result<Self, DeclarativeRegistryError> {
        let mut shapes = Vec::with_capacity(operation_shapes.len());
        for &(name, shape) in operation_shapes {
            if name.is_empty() {
                return Err(DeclarativeRegistryError::EmptyOperation);
            }
            if CORE_OPERATIONS
                .iter()
                .any(|descriptor| descriptor.name == name)
                || name == "module"
            {
                return Err(DeclarativeRegistryError::CoreOperation(name.to_owned()));
            }
            if shapes.iter().any(|(candidate, _)| candidate == name) {
                return Err(DeclarativeRegistryError::DuplicateOperation(
                    name.to_owned(),
                ));
            }
            shapes.push((name.to_owned(), shape));
        }
        Ok(Self {
            operations: CORE_OPERATIONS,
            types: &[],
            attributes: &[],
            operation_shapes: Some(shapes.into_boxed_slice()),
            module_alias: true,
        })
    }

    pub fn type_descriptor(&self, name: &str) -> Option<&TypeDescriptor> {
        self.types.iter().find(|descriptor| descriptor.name == name)
    }

    pub fn attribute_descriptor(&self, name: &str) -> Option<&AttributeDescriptor> {
        self.attributes
            .iter()
            .find(|descriptor| descriptor.name == name)
    }

    pub fn region(&self, operation: &str, index: usize) -> RegionDescriptor {
        self.operation(operation)
            .and_then(|descriptor| descriptor.regions.get(index))
            .copied()
            .unwrap_or_default()
    }

    pub fn symbols(&self, operation: &str) -> SymbolDescriptor {
        self.operation(operation)
            .map(|descriptor| descriptor.symbols)
            .unwrap_or_default()
    }

    pub fn proving() -> &'static Self {
        &PROVING_REGISTRY
    }

    /// Returns the standard module and function operation set.
    pub fn core() -> &'static Self {
        &CORE_REGISTRY
    }

    pub(crate) fn custom_operation(&self, spelling: &str) -> Option<&OperationDescriptor> {
        self.operation(spelling).or_else(|| {
            (spelling == "module" && (self.module_alias || std::ptr::eq(self, &CORE_REGISTRY)))
                .then(|| self.operation("builtin.module"))
                .flatten()
        })
    }

    /// Builds a callback-free registry set from the built-in declarative operation catalog.
    ///
    /// Iteration follows the fixed proving-catalog order, independent of input order.
    pub fn declarative(operation_names: &[&str]) -> Result<Self, DeclarativeRegistryError> {
        let mut selected = 0_u8;
        for &name in operation_names {
            let index = PROVING_OPERATIONS
                .iter()
                .position(|descriptor| descriptor.name == name)
                .ok_or_else(|| DeclarativeRegistryError::UnknownOperation(name.to_owned()))?;
            let bit = 1_u8 << index;
            if selected & bit != 0 {
                return Err(DeclarativeRegistryError::DuplicateOperation(
                    name.to_owned(),
                ));
            }
            selected |= bit;
        }
        let operations: &'static [OperationDescriptor] =
            DECLARATIVE_OPERATION_SETS[usize::from(selected)].get_or_init(|| {
                PROVING_OPERATIONS
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| selected & (1_u8 << index) != 0)
                    .map(|(_, descriptor)| *descriptor)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            });
        Ok(Self {
            operations,
            types: &[],
            attributes: &[],
            operation_shapes: None,
            module_alias: false,
        })
    }

    pub(crate) fn content_identity(&self) -> u64 {
        fn mix(mut hash: u64, bytes: &[u8]) -> u64 {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash
        }

        let mut hash = 0xcbf29ce484222325;
        for descriptor in self.operations {
            hash = mix(hash, descriptor.name.as_bytes());
            hash = mix(hash, &[descriptor.symbols.defines_symbol as u8]);
            hash = mix(hash, &[descriptor.symbols.symbol_table as u8]);
            hash = mix(hash, &[descriptor.symbols.uses_symbols as u8]);
            hash = mix(hash, &(descriptor.regions.len() as u64).to_le_bytes());
            for region in descriptor.regions {
                hash = mix(hash, &[matches!(region.kind, RegionKind::Graph) as u8]);
                hash = mix(hash, &[region.isolated_from_above as u8]);
            }
        }
        if let Some(shapes) = &self.operation_shapes {
            for (name, shape) in shapes.iter() {
                hash = mix(hash, name.as_bytes());
                hash = mix(hash, &[*shape as u8]);
            }
        }
        for descriptor in self.types {
            hash = mix(hash, descriptor.name.as_bytes());
        }
        for descriptor in self.attributes {
            hash = mix(hash, descriptor.name.as_bytes());
        }
        hash
    }
}

fn lower_arith_constant(context: &RegisteredLoweringContext<'_>) -> Option<RegisteredLowering> {
    let spelling = context.spelling();
    let (_, tail) = spelling.split_once("arith.constant")?;
    let tail = tail.trim();
    let (value, ty) = tail.rsplit_once(':')?;
    let value = value.split('{').next().unwrap_or(value).trim();
    let ty = ty.trim();
    let compatible = if value.parse::<i128>().is_ok() {
        ty == "index"
            || ty
                .strip_prefix('i')
                .is_some_and(|width| width.parse::<u32>().is_ok())
            || ty
                .strip_prefix("si")
                .or_else(|| ty.strip_prefix("ui"))
                .is_some_and(|width| width.parse::<u32>().is_ok())
    } else if value.parse::<f64>().is_ok() {
        ty.strip_prefix('f')
            .is_some_and(|width| width.parse::<u32>().is_ok())
    } else {
        true
    };
    (!value.is_empty() && !ty.is_empty()).then(|| RegisteredLowering {
        name: "arith.constant",
        result_types: compatible.then(|| ty.to_owned()).into_iter().collect(),
        function_type: format!("() -> {ty}"),
        attributes: vec![("value", value.to_owned())],
    })
}

fn lower_module(context: &RegisteredLoweringContext<'_>) -> Option<RegisteredLowering> {
    let mut attributes = Vec::new();
    if let Some(symbol) = context.leading_symbol() {
        attributes.push(("sym_name", symbol.to_owned()));
    }
    Some(RegisteredLowering {
        name: "builtin.module",
        result_types: Vec::new(),
        function_type: "() -> ()".into(),
        attributes,
    })
}

fn argument_type<'a>(spelling: &'a str, attributes: Option<&str>) -> Option<&'a str> {
    let (_, ty) = spelling.split_once(':')?;
    let end = attributes
        .and_then(|attributes| spelling.rfind(attributes))
        .or_else(|| {
            ty.find(" loc(")
                .map(|offset| spelling.len() - ty.len() + offset)
        })
        .unwrap_or(spelling.len());
    Some(spelling[spelling.len() - ty.len()..end].trim())
}

fn function_signature(context: &RegisteredLoweringContext<'_>) -> Option<String> {
    let inputs = context
        .arguments()
        .map(|(argument, attributes)| argument_type(argument, attributes))
        .collect::<Option<Vec<_>>>()?
        .join(", ");
    let results = if let Some(tail) = context.function_results() {
        let tail = tail.strip_prefix("->")?.trim_start();
        let tail = tail.trim_start();
        let raw = if tail.starts_with('(') {
            let end = matching_delimiter(tail, 0, '(', ')')?;
            &tail[..=end]
        } else {
            tail.split_whitespace().next().unwrap_or(tail).trim()
        };
        if raw.starts_with('(') {
            let inner = &raw[1..raw.len() - 1];
            let types = split_top_level(inner)
                .iter()
                .map(|result| strip_top_level_attribute(result).trim())
                .collect::<Vec<_>>();
            format!("({})", types.join(", "))
        } else {
            strip_top_level_attribute(raw).trim().to_owned()
        }
    } else {
        "()".to_owned()
    };
    Some(format!("({inputs}) -> {results}"))
}

fn split_top_level(value: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    let mut quoted = false;
    let mut escaped = false;
    for (index, character) in value.char_indices() {
        if quoted {
            if character == '"' && !escaped {
                quoted = false;
            }
            escaped = character == '\\' && !escaped;
            if character != '\\' {
                escaped = false;
            }
            continue;
        }
        match character {
            '"' => quoted = true,
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            '>' if depth > 0 => depth -= 1,
            ',' if depth == 0 => {
                result.push(value[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    if !value[start..].trim().is_empty() {
        result.push(value[start..].trim());
    }
    result
}

fn attribute_groups<'a>(groups: impl IntoIterator<Item = Option<&'a str>>) -> Option<String> {
    let groups = groups
        .into_iter()
        .flatten()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    (!groups.is_empty()).then(|| format!("[{}]", groups.join(", ")))
}

fn strip_top_level_attribute(value: &str) -> &str {
    top_level_attribute(value)
        .map(|(start, _)| &value[..start])
        .unwrap_or(value)
}

fn top_level_attribute(value: &str) -> Option<(usize, &str)> {
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    for (index, character) in value.char_indices() {
        if quoted {
            if character == '"' && !escaped {
                quoted = false;
            }
            escaped = character == '\\' && !escaped;
            if character != '\\' {
                escaped = false;
            }
            continue;
        }
        match character {
            '"' => quoted = true,
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth = depth.saturating_sub(1),
            '{' if depth == 0 => {
                let end = matching_delimiter(value, index, '{', '}')?;
                return Some((index, &value[index..=end]));
            }
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    None
}

fn matching_delimiter(spelling: &str, open: usize, left: char, right: char) -> Option<usize> {
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    for (offset, character) in spelling[open..].char_indices() {
        if quoted {
            if character == '"' && !escaped {
                quoted = false;
            }
            escaped = character == '\\' && !escaped;
            if character != '\\' {
                escaped = false;
            }
            continue;
        }
        if character == '"' {
            quoted = true;
            continue;
        }
        if character == left {
            depth += 1;
        } else if character == right {
            depth -= 1;
            if depth == 0 {
                return Some(open + offset);
            }
        }
    }
    None
}

fn lower_function(context: &RegisteredLoweringContext<'_>) -> Option<RegisteredLowering> {
    lower_func_like("func.func", context)
}

fn lower_func_like(
    _operation: &str,
    context: &RegisteredLoweringContext<'_>,
) -> Option<RegisteredLowering> {
    let symbol = context.leading_symbol()?.to_owned();
    let signature = function_signature(context)?;
    let mut attributes = vec![("sym_name", symbol), ("function_type", signature)];
    if let Some(visibility @ ("public" | "private" | "nested")) = context.visibility() {
        attributes.push(("sym_visibility", format!("\"{visibility}\"")));
    }
    if let Some(groups) = attribute_groups(context.arguments().map(|(_, attributes)| attributes)) {
        attributes.push(("arg_attrs", groups));
    }
    if let Some(results) = context.function_results() {
        let results = results.strip_prefix("->")?.trim();
        let groups = split_top_level(
            results
                .strip_prefix('(')
                .and_then(|value| value.strip_suffix(')'))
                .unwrap_or(results),
        )
        .into_iter()
        .map(|result| top_level_attribute(result).map(|(_, group)| group));
        if let Some(groups) = attribute_groups(groups) {
            attributes.push(("res_attrs", groups));
        }
    }
    Some(RegisteredLowering {
        name: "func.func",
        result_types: Vec::new(),
        function_type: "() -> ()".into(),
        attributes,
    })
}

fn lower_call(context: &RegisteredLoweringContext<'_>) -> Option<RegisteredLowering> {
    lower_call_like("func.call", context)
}

fn lower_call_like(
    _operation: &str,
    context: &RegisteredLoweringContext<'_>,
) -> Option<RegisteredLowering> {
    let callee = context.leading_symbol()?.to_owned();
    let tail = context.function_type()?;
    let (inputs, results) = tail.split_once("->")?;
    let result_types = crate::semantic::split_registered_types(results);
    Some(RegisteredLowering {
        name: "func.call",
        result_types,
        function_type: format!("{} -> {}", inputs.trim(), results.trim()),
        attributes: vec![("callee", callee)],
    })
}

pub(crate) fn lower_operation_shape(
    shape: OperationShape,
    operation: &str,
    context: &RegisteredLoweringContext<'_>,
) -> Option<RegisteredLowering> {
    match shape {
        OperationShape::FuncLike => lower_func_like(operation, context),
        OperationShape::CallLike => lower_call_like(operation, context),
    }
}

fn verify_arith_constant(document: &Document, operation: OperationId) -> Result<(), &'static str> {
    let results = document
        .result_types(operation)
        .ok_or("missing result list")?;
    if !document
        .operands(operation)
        .is_some_and(|values| values.is_empty())
        || results.len() != 1
    {
        return Err("arith.constant expects zero operands and one result");
    }
    let value = document
        .attribute_id(operation, "value")
        .ok_or("arith.constant requires `value`")?;
    let result = document
        .type_value(results[0])
        .ok_or("arith.constant has an invalid result type")?;
    match (document.attribute_value(value), result) {
        (
            Some(crate::semantic::AttributeValue::Integer(_)),
            crate::semantic::TypeValue::Integer { .. } | crate::semantic::TypeValue::Index,
        )
        | (Some(crate::semantic::AttributeValue::Float(_)), crate::semantic::TypeValue::Float(_)) =>
            {}
        (Some(crate::semantic::AttributeValue::Integer(_)), _) => {
            return Err("arith.constant integer value requires an integer or index result");
        }
        (Some(crate::semantic::AttributeValue::Float(_)), _) => {
            return Err("arith.constant floating-point value requires a floating-point result");
        }
        _ => return Err("arith.constant value must be an integer or floating-point attribute"),
    }
    Ok(())
}

fn print_attribute_dictionary(
    document: &Document,
    operation: OperationId,
    excluded: &[&str],
) -> Option<String> {
    let entries = document
        .attributes(operation)?
        .filter(|(name, _)| !excluded.contains(name))
        .map(|(name, value)| format!("{name} = {value}"))
        .collect::<Vec<_>>();
    Some(if entries.is_empty() {
        String::new()
    } else {
        format!(" {{{}}}", entries.join(", "))
    })
}

fn print_arith_constant(document: &Document, operation: OperationId) -> Option<String> {
    let value = document
        .attributes(operation)?
        .find_map(|(name, value)| (name == "value").then_some(value))?;
    let ty = document
        .result_types(operation)?
        .first()
        .and_then(|ty| document.type_spelling(*ty))?;
    let dictionary = print_attribute_dictionary(document, operation, &["value"])?;
    Some(format!("arith.constant {value}{dictionary} : {ty}"))
}

fn lower_addi(context: &RegisteredLoweringContext<'_>) -> Option<RegisteredLowering> {
    let tail = context.spelling().split_once("arith.addi")?.1;
    let (head, ty) = tail.rsplit_once(':')?;
    let ty = ty.trim();
    let attributes = head
        .find("overflow")
        .and_then(|start| {
            let value = &head[start..];
            let open = value.find('<')?;
            let close = value[open + 1..].find('>')? + open + 1;
            let flags = strip_overflow_trivia(&value[open + 1..close]);
            Some(vec![("overflowFlags", format!("#arith.overflow<{flags}>"))])
        })
        .unwrap_or_default();
    Some(RegisteredLowering {
        name: "arith.addi",
        result_types: vec![ty.into()],
        function_type: format!("({ty}, {ty}) -> {ty}"),
        attributes,
    })
}

fn strip_overflow_trivia(value: &str) -> String {
    value
        .lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .flat_map(str::chars)
        .filter(|character| !character.is_ascii_whitespace())
        .collect()
}

fn lower_no_results(name: &'static str) -> Option<RegisteredLowering> {
    Some(RegisteredLowering {
        name,
        result_types: Vec::new(),
        function_type: "() -> ()".into(),
        attributes: Vec::new(),
    })
}

fn lower_return(context: &RegisteredLoweringContext<'_>) -> Option<RegisteredLowering> {
    let tail = context.spelling().split_once("func.return")?.1.trim();
    let input = tail
        .rsplit_once(':')
        .map(|(_, types)| types.trim())
        .unwrap_or("()");
    let input = if input.starts_with('(') {
        input.to_owned()
    } else {
        format!("({input})")
    };
    Some(RegisteredLowering {
        name: "func.return",
        result_types: Vec::new(),
        function_type: format!("{input} -> ()"),
        attributes: Vec::new(),
    })
}

fn verify_addi(document: &Document, operation: OperationId) -> Result<(), &'static str> {
    let expected = document
        .result_types(operation)
        .and_then(|types| types.first())
        .and_then(|ty| document.type_spelling(*ty))
        .ok_or("arith.addi requires one result")?;
    if document
        .operands(operation)
        .ok_or("arith.addi requires operands")?
        .iter()
        .any(|operand| document.value_type(*operand) != Some(expected))
    {
        return Err("arith.addi operand and result types must match");
    }
    if let Some(flags) = document.attributes(operation).and_then(|mut attrs| {
        attrs.find_map(|(name, value)| (name == "overflowFlags").then_some(value))
    }) {
        if !matches!(
            flags,
            "#arith.overflow<none>"
                | "#arith.overflow<nsw>"
                | "#arith.overflow<nuw>"
                | "#arith.overflow<nsw,nuw>"
                | "#arith.overflow<nuw,nsw>"
        ) {
            return Err("arith.addi has unrecognized overflow flags");
        }
    }
    Ok(())
}

fn print_addi(document: &Document, operation: OperationId) -> Option<String> {
    let operands = document.operands(operation)?;
    let ty = document
        .result_types(operation)?
        .first()
        .and_then(|ty| document.type_spelling(*ty))?;
    let flags = document
        .attributes(operation)?
        .find_map(|(name, value)| {
            (name == "overflowFlags").then(|| value.trim_start_matches("#arith.").to_owned())
        })
        .map(|value| format!(" {value}"))
        .unwrap_or_default();
    let dictionary = print_attribute_dictionary(document, operation, &["overflowFlags"])?;
    Some(format!(
        "arith.addi {}, {}{flags}{dictionary} : {ty}",
        document.value_spelling(operands[0])?,
        document.value_spelling(operands[1])?
    ))
}

fn print_return(document: &Document, operation: OperationId) -> Option<String> {
    let operands = document.typed_operands_spelling(operation)?;
    let dictionary = print_attribute_dictionary(document, operation, &[])?;
    let body = operands
        .find(" :")
        .map(|colon| format!("{}{dictionary}{}", &operands[..colon], &operands[colon..]))
        .unwrap_or_else(|| format!("{operands}{dictionary}"));
    Some(format!("func.return{body}"))
}

fn print_branch(document: &Document, operation: OperationId) -> Option<String> {
    Some(format!(
        "cf.br {}{}",
        document.successor_spelling(operation)?,
        print_attribute_dictionary(document, operation, &[])?
    ))
}

fn print_module(document: &Document, operation: OperationId) -> Option<String> {
    let symbol = document
        .attributes(operation)?
        .find_map(|(name, value)| (name == "sym_name").then_some(value));
    let dictionary = print_attribute_dictionary(document, operation, &["sym_name"])?;
    Some(format!(
        "builtin.module{}{} ",
        symbol.map(|value| format!(" {value}")).unwrap_or_default(),
        if dictionary.is_empty() {
            String::new()
        } else {
            format!(" attributes{dictionary}")
        }
    ))
}

fn print_function(document: &Document, operation: OperationId) -> Option<String> {
    let symbol = document
        .attributes(operation)?
        .find_map(|(name, value)| (name == "sym_name").then_some(value))?;
    let signature = document
        .attributes(operation)?
        .find_map(|(name, value)| (name == "function_type").then_some(value))?;
    let signature = signature
        .strip_prefix("type<")
        .and_then(|value| value.strip_suffix('>'))
        .unwrap_or(signature);
    let no_inline = document
        .attribute_id(operation, "no_inline")
        .and_then(|id| document.attribute_value(id))
        .is_some_and(is_unit_attribute);
    let dictionary = print_attribute_dictionary(
        document,
        operation,
        &["sym_name", "function_type", "sym_visibility", "no_inline"],
    )?;
    let dictionary = if no_inline {
        if dictionary.is_empty() {
            " {no_inline}".to_owned()
        } else {
            format!("{}{}, no_inline}}", &dictionary[..dictionary.len() - 1], "")
        }
    } else {
        dictionary
    };
    let visibility = document
        .attributes(operation)?
        .find_map(|(name, value)| (name == "sym_visibility").then_some(value.trim_matches('"')))
        .map(|value| format!(" {value}"))
        .unwrap_or_default();
    Some(format!(
        "func.func{visibility} {symbol}{signature}{} ",
        if dictionary.is_empty() {
            String::new()
        } else {
            format!(" attributes{dictionary}")
        }
    ))
}

fn is_unit_attribute(value: &crate::semantic::AttributeValue) -> bool {
    matches!(value, crate::semantic::AttributeValue::Opaque(bytes) if bytes.as_ref() == b"unit")
}

fn print_call(document: &Document, operation: OperationId) -> Option<String> {
    let callee = document
        .attributes(operation)?
        .find_map(|(name, value)| (name == "callee").then_some(value))?;
    let operands = document
        .operands(operation)?
        .iter()
        .map(|operand| document.value_spelling(*operand))
        .collect::<Option<Vec<_>>>()?
        .join(", ");
    let dictionary = print_attribute_dictionary(document, operation, &["callee"])?;
    Some(format!(
        "func.call {callee}({operands}){dictionary} : {}",
        document.type_spelling(document.function_type(operation)?)?
    ))
}

fn print_cond_branch(document: &Document, operation: OperationId) -> Option<String> {
    let operands = document.operands(operation)?;
    let condition = document.value_spelling(*operands.first()?)?;
    let successors = document.successors(operation)?;
    if successors.len() != 2 {
        return None;
    }
    let dictionary = print_attribute_dictionary(document, operation, &[])?;
    Some(format!(
        "cf.cond_br {condition}, {}, {}{dictionary}",
        document.successor_spelling_at(operation, 0)?,
        document.successor_spelling_at(operation, 1)?
    ))
}

static MODULE_REGIONS: &[RegionDescriptor] = &[RegionDescriptor {
    kind: RegionKind::Ssacfg,
    isolated_from_above: true,
}];
static FUNCTION_REGIONS: &[RegionDescriptor] = &[RegionDescriptor {
    kind: RegionKind::Ssacfg,
    isolated_from_above: true,
}];

static BUILTIN_MODULE: OperationDescriptor = OperationDescriptor {
    name: "builtin.module",
    syntax_kind: SyntaxKind::DialectOperation,
    parse: None,
    lower: None,
    verify: None,
    print: None,
    assembly: Some(AssemblyProgram::Module),
    schema: OperationSchema {
        operands: OperandCount::Exact(0),
        results: ResultCount::Exact(0),
        required_attributes: &[],
    },
    regions: MODULE_REGIONS,
    symbols: SymbolDescriptor {
        defines_symbol: true,
        symbol_table: true,
        uses_symbols: false,
    },
};

static FUNC_FUNC: OperationDescriptor = OperationDescriptor {
    name: "func.func",
    syntax_kind: SyntaxKind::DialectOperation,
    parse: None,
    lower: None,
    verify: None,
    print: None,
    assembly: Some(AssemblyProgram::Function),
    schema: OperationSchema {
        operands: OperandCount::Exact(0),
        results: ResultCount::Exact(0),
        required_attributes: &["sym_name", "function_type"],
    },
    regions: FUNCTION_REGIONS,
    symbols: SymbolDescriptor {
        defines_symbol: true,
        symbol_table: false,
        uses_symbols: false,
    },
};

static FUNC_CALL: OperationDescriptor = OperationDescriptor {
    name: "func.call",
    syntax_kind: SyntaxKind::DialectOperation,
    parse: None,
    lower: None,
    verify: None,
    print: None,
    assembly: Some(AssemblyProgram::Call),
    schema: OperationSchema {
        operands: OperandCount::Variadic,
        results: ResultCount::Variadic,
        required_attributes: &["callee"],
    },
    regions: &[],
    symbols: SymbolDescriptor {
        defines_symbol: false,
        symbol_table: false,
        uses_symbols: true,
    },
};

static CF_COND_BR: OperationDescriptor = OperationDescriptor {
    name: "cf.cond_br",
    syntax_kind: SyntaxKind::DialectOperation,
    parse: None,
    lower: None,
    verify: None,
    print: None,
    assembly: Some(AssemblyProgram::ConditionalBranch),
    schema: OperationSchema {
        operands: OperandCount::Variadic,
        results: ResultCount::Exact(0),
        required_attributes: &[],
    },
    regions: &[],
    symbols: SymbolDescriptor {
        defines_symbol: false,
        symbol_table: false,
        uses_symbols: false,
    },
};

static ARITH_CONSTANT: OperationDescriptor = OperationDescriptor {
    name: "arith.constant",
    syntax_kind: SyntaxKind::DialectOperation,
    parse: None,
    lower: None,
    verify: None,
    print: None,
    assembly: Some(AssemblyProgram::TypedAttribute),
    schema: OperationSchema {
        operands: OperandCount::Exact(0),
        results: ResultCount::Exact(1),
        required_attributes: &["value"],
    },
    regions: &[],
    symbols: SymbolDescriptor {
        defines_symbol: false,
        symbol_table: false,
        uses_symbols: false,
    },
};

static ARITH_ADDI: OperationDescriptor = OperationDescriptor {
    name: "arith.addi",
    syntax_kind: SyntaxKind::DialectOperation,
    parse: None,
    lower: None,
    verify: None,
    print: None,
    assembly: Some(AssemblyProgram::BinaryOperands),
    schema: OperationSchema {
        operands: OperandCount::Exact(2),
        results: ResultCount::Exact(1),
        required_attributes: &[],
    },
    regions: &[],
    symbols: SymbolDescriptor {
        defines_symbol: false,
        symbol_table: false,
        uses_symbols: false,
    },
};
static FUNC_RETURN: OperationDescriptor = OperationDescriptor {
    name: "func.return",
    syntax_kind: SyntaxKind::DialectOperation,
    parse: None,
    lower: None,
    verify: None,
    print: None,
    assembly: Some(AssemblyProgram::OptionalTypedOperands),
    schema: OperationSchema {
        operands: OperandCount::Variadic,
        results: ResultCount::Exact(0),
        required_attributes: &[],
    },
    regions: &[],
    symbols: SymbolDescriptor {
        defines_symbol: false,
        symbol_table: false,
        uses_symbols: false,
    },
};
static CF_BR: OperationDescriptor = OperationDescriptor {
    name: "cf.br",
    syntax_kind: SyntaxKind::DialectOperation,
    parse: None,
    lower: None,
    verify: None,
    print: None,
    assembly: Some(AssemblyProgram::TypedSuccessor),
    schema: OperationSchema {
        operands: OperandCount::Exact(0),
        results: ResultCount::Exact(0),
        required_attributes: &[],
    },
    regions: &[],
    symbols: SymbolDescriptor {
        defines_symbol: false,
        symbol_table: false,
        uses_symbols: false,
    },
};
static PROVING_OPERATIONS: &[OperationDescriptor] = &[
    BUILTIN_MODULE,
    FUNC_FUNC,
    FUNC_RETURN,
    FUNC_CALL,
    ARITH_CONSTANT,
    ARITH_ADDI,
    CF_BR,
    CF_COND_BR,
];
static CORE_OPERATIONS: &[OperationDescriptor] =
    &[BUILTIN_MODULE, FUNC_FUNC, FUNC_RETURN, FUNC_CALL];
static DECLARATIVE_OPERATION_SETS: [OnceLock<Box<[OperationDescriptor]>>; 256] =
    [const { OnceLock::new() }; 256];
static PROVING_REGISTRY: DialectRegistry = DialectRegistry::new(PROVING_OPERATIONS, &[], &[]);
static CORE_REGISTRY: DialectRegistry = DialectRegistry::new(CORE_OPERATIONS, &[], &[]);
