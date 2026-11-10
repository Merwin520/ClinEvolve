from pathlib import Path
import importlib

def load_tools(logging=print, names=[]):
    tools_dir = Path(__file__).parent
    tools = []

    # Get all Python files in the tools directory (excluding __init__.py)
    tool_files = [
        f
        for f in tools_dir.glob("*.py")
        if f.stem != "__init__"
    ]

    # If the caller supplied an explicit tool list, preserve that
    # exact order rather than inheriting filesystem/glob order.
    if names and names != "all":
        files_by_name = {
            f.stem: f
            for f in tool_files
        }

        tool_files = [
            files_by_name[name]
            for name in dict.fromkeys(names)
            if name in files_by_name
        ]

    for tool_file in tool_files:
        module_name = f"agent.tools.{tool_file.stem}"

        try:
            module = importlib.import_module(module_name)

            if (
                hasattr(module, "tool_info")
                and hasattr(module, "tool_function")
            ):
                tool_name = tool_file.stem

                if names and (
                    names == "all"
                    or tool_name in names
                ):
                    tools.append({
                        "info": module.tool_info(),
                        "function": module.tool_function,
                        "name": tool_name,
                    })
            else:
                raise Exception(
                    f"Tool module {module_name} "
                    "does not have required functions."
                )

        except Exception as e:
            logging(
                f"Failed to import {module_name}: {e}"
            )
            raise

    return tools
