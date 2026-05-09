# OpenSpiel functions: registered_names

[Back to Core API reference](../api_reference.md) \
<br>

`registered_names()`

Returns a list of short names of all games in the library. These are names that
can be used when loading games in `load_game`.

In this trimmed build, the returned list is:

```python
["carcassonne", "othello", "tic_tac_toe"]
```

## Examples:

```python
import pyspiel

# Print the name of all games in this build
for short_name in pyspiel.registered_names():
  print(short_name)
```
