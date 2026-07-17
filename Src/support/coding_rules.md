# Coding Rules and Style Guide

This document defines the coding standards, naming conventions, and design rules for the `support` module files.

## 1. Naming Conventions

### Classes
* **Style:** `PascalCase`
* **Examples:** `Drawer`, `HumanDetector`, `BoundingBox`, `SegmentedObject`, `ShellUtil`
* **Exception:** Legacy classes (e.g. `SRC_INFO`) are maintained for backward compatibility but new classes must use `PascalCase`.

### Class Attributes & Fields
* **Style:** Lowercase `snake_case`
* **Examples:** `class_id`, `confidence`, `roi_points`, `label_color`
* *Note:* Attributes must not start with capital letters (e.g., `Type` or `Roi_color` are incorrect and must be refactored to `type` and `roi_color`).

### Functions & Methods
* **Style:** Lowercase `snake_case`
* **Examples:** `draw()`, `track_human()`, `mask_shirt()`, `read_file()`

### Local Variables & Parameters
* **Style:** Lowercase `snake_case`
* **Examples:** `frame_width`, `log_level`, `detected_human`, `x1`, `y1`

### Constants & Global Variables
* **Style:** Uppercase `SCREAMING_SNAKE_CASE`
* **Examples:** `INTERNAL_MODEL_PATH`, `MODULE_DIR`, `TARGET_TOP_CLASSES`

---

## 2. Function Visibility & Access (Internal vs. External)

### Internal Functions/Methods (Private/Protected)
* **Rule:** Must be prefixed with a single leading underscore `_`.
* **Usage:** Only callable from within the same class or module.
* **Examples:** `_segment_shirt()`, `_parse_unc_path()`, `_get_absolute_path()`, `_update()`, `_real_log_debug()`

### External Functions/Methods (Public API)
* **Rule:** Must NOT have a leading underscore.
* **Usage:** Safe to be called from other modules/classes.
* **Examples:** `draw()`, `track_human()`, `detect_human()`, `mask_shirt()`, `start()`, `read()`, `stop()`
* **CRITICAL:** Do NOT change the prototypes (signatures) of external functions to prevent breaking code calling them from outside the module.

---

## 3. Spelling & Typography Guidelines

* Always use the correct spelling of common words:
  * Use `label` (e.g. `label_color`, `label_point`), never `lable` (e.g. `Lable_color`).
  * Use `length` (e.g. logging messages), never `lenght`.
* Ensure consistent whitespace around operators and after commas in function arguments.

---

## 4. Type Hinting Guidelines

* Use standard Python type hinting:
  * Use `np.ndarray` for NumPy arrays instead of `np.array` in type declarations.
  * Use proper generic collection types (e.g., `list[BoundingBox]`, `tuple[bool, np.ndarray]`).
  * Return type annotations must match the actual return type (e.g., if a method returns a list of objects, it should be annotated with `list[ObjectType]` rather than `ObjectType`).
