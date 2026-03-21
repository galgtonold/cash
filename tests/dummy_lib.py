
# We need a shared Cash instance if we want them to share the registry?
# Or does each module have its own Cash instance?
# Usually users create one 'app = Cash()' and import it, or create it in main.
# If I create 'lib_app = Cash()' here, it's a different registry.
# If I want 'main_app' to know about 'lib_func', they need to share the registry?
# Or 'main_app' needs to know about 'lib_func' via import?

# If I use a different app instance, 'main_app' won't know 'lib_func' is cached.
# So 'main_app' won't track it as a dependency.
# This is a design question.
# Usually in Flask/Celery, you have a single app instance.
# So let's assume the user passes the app or imports it.

# For testing, let's define a function that takes an app?
# Or just expose a function and let the test decorate it?

def lib_func(x):
    return x + 10
