import sys
import importlib
import os

def main():
    if len(sys.argv) < 2:
        print("Uso: anmetal <comando> [args]")
        print("Comandos disponibles:")
        # Listar archivos .py en la carpeta commands
        commands_dir = os.path.join(os.path.dirname(__file__), 'commands')
        if os.path.exists(commands_dir):
            for f in os.listdir(commands_dir):
                if f.endswith('.py') and not f.startswith('__'):
                    print(f"  - {f[:-3]}")
        sys.exit(1)

    command_name = sys.argv[1]
    args = sys.argv[2:]

    # Modificar sys.argv para que el script importado vea los argumentos correctos
    sys.argv = [f"anmetal {command_name}"] + args

    try:
        # Intenta importar el módulo desde anmetal.commands.<comando>
        module = importlib.import_module(f"anmetal.commands.{command_name}")
        
        # Si el módulo tiene una función main, la ejecutamos.
        if hasattr(module, 'main'):
            module.main()
            
    except ModuleNotFoundError:
        print(f"Error: El comando '{command_name}' no existe.")
        sys.exit(1)
    except Exception as e:
        print(f"Error al ejecutar el comando '{command_name}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
