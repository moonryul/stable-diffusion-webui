import sys

from modules.shared_cmd_options import cmd_opts


def realesrgan_models_names():
    import modules.realesrgan_model
    return [x.name for x in modules.realesrgan_model.get_realesrgan_models(None)]


def postprocessing_scripts():
    import modules.scripts

    return modules.scripts.scripts_postproc.scripts


def sd_vae_items():
    import modules.sd_vae

    return ["Automatic", "None"] + list(modules.sd_vae.vae_dict)


def refresh_vae_list():
    import modules.sd_vae

    modules.sd_vae.refresh_vae_list()


def cross_attention_optimizations():
    import modules.sd_hijack

    return ["Automatic"] + [x.title() for x in modules.sd_hijack.optimizers] + ["None"]


def sd_unet_items():
    import modules.sd_unet

    return ["Automatic"] + [x.label for x in modules.sd_unet.unet_options] + ["None"]


def refresh_unet_list():
    import modules.sd_unet

    modules.sd_unet.list_unets()


def list_checkpoint_tiles():
    import modules.sd_models
    return modules.sd_models.checkpoint_tiles()


def refresh_checkpoints():
    import modules.sd_models
    return modules.sd_models.list_models()


def list_samplers():
    import modules.sd_samplers
    return modules.sd_samplers.all_samplers


def reload_hypernetworks():
    from modules.hypernetworks import hypernetwork
    from modules import shared

    shared.hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)


ui_reorder_categories_builtin_items = [
    "inpaint",
    "sampler",
    "accordions",
    "checkboxes",
    "dimensions",
    "cfg",
    "denoising",
    "seed",
    "batch",
    "override_settings",
]


def ui_reorder_categories():
    from modules import scripts

    yield from ui_reorder_categories_builtin_items

    sections = {}
    for script in scripts.scripts_txt2img.scripts + scripts.scripts_img2img.scripts:
        if isinstance(script.section, str) and script.section not in ui_reorder_categories_builtin_items:
            sections[script.section] = 1

    yield from sections

    yield "scripts"

#MJ: sys.modules[__name__].__class__ is class "module"
class Shared( sys.modules[__name__].__class__ ):
    #MJ; __name__ ="shared_items": Shared is a child class of sys.modules['shared_items"].__class__ 
    # Shared is not a new module or submodule.  The term "submodule" in Python usually refers to a module inside a package.
    # Instead, it's a subclass of the module class that introduces additional behavior 
    # to an existing module (modules.shared).
    """
    this class is here to provide **sd_model field** as a property, so that it can be created and loaded on demand rather than
    at program startup.
    """

    sd_model_val = None

    @property
    def sd_model(self):  #MJ: Mentining modules.shared.sd_model invokes this get function; e.g. by shared.sd_model property mentioned in load_model, the target function of the thread lauched in initialize_rest()
        import modules.sd_models

        return modules.sd_models.model_data.get_sd_model()
        #MJ:  return self.sd_model from  modules.sd_models.#MJ: self.sd_model was  set by  model_data.set_sd_model(sd_model) in load_model() 


    @sd_model.setter
    def sd_model(self, value):
        import modules.sd_models

        modules.sd_models.model_data.set_sd_model(value)

#MJ: After defining this subclass, the line sys.modules['modules.shared'].__class__ = Shared 
# is effectively changing the class (type) of the modules.shared module to be this new Shared class.

# The code is taking advantage of Python's dynamic nature to modify the behavior of the modules.shared module at runtime.
# After this modification, whenever someone accesses the sd_model attribute of the modules.shared module, 
# it will invoke the getter (and potentially setter) of the sd_model property defined in the Shared class.

# However, all other functionalities of the original module class would remain intact
# unless they are explicitly overridden in the Shared subclass. 
# In your provided code, only the sd_model property is added,
# and there's no indication that any of the original module class's behaviors are overridden.

sys.modules['modules.shared'].__class__ = Shared  

# This is described by Guido van Rossum as a hack that is sometimes recommended:

# https://mail.python.org/pipermail/python-ideas/2012-May/014969.html

# # foo.py
# import sys

# class Foo:
#     def funct1(self, <args>): <code>
#     def funct2(self, <args>): <code>

# sys.modules[__name__] = Foo()

# You can easily override __getattr__ / __getattribute__ / __setattr__
#this way. It also makes "subclassing" the module a little easier
# (although accessing the class to be used as a base class is a little
# tricky: you'd have to use foo.__class__). 

# In the vast, vast  majority of cases it's better to clearly have a separation between
# modules, which provide no inheritance and no instantiation, and
# classes, which provide both. I think Python is better off this way
# than Java, where all you have is classes (its packages cannot contain
# anything except class definitions).



#==> so: modules.shared.sd_model becomes Shared.sd_model which holds an model of LatentDiffusion, 
# with its get and set methods

# In essence, the modules.shared module will take on the behavior and attributes of the Shared class, 
# which may include any properties or methods defined in that class. This can lead to customized 
# and non-standard behavior for the module.


#MJ: shared.sd_model property trace:

# (1) 
# Module "modules.shared.py" defines  a field "sd_model" as follows:


# sd_model: sd_models_types.WebuiSdModel = None
# in modules.shared.py.

# (2)  modules.shared is modified as follows:

# lass Shared( sys.modules[__name__].__class__ ): #MJ; __name__ ="shared_items": Shared is a child class of sys.modules['shared_items"].__class__ 
#     """
#     this class is here to provide **sd_model field** as a property, so that it can be created and loaded on demand rather than
#     at program startup.
#     """

#     sd_model_val = None

#     @property
#     def sd_model(self):  #MJ: modules.shared.sd_model ==>  invokes this get function
#         import modules.sd_models

#         return modules.sd_models.model_data.get_sd_model()

#     @sd_model.setter
#     def sd_model(self, value):
#         import modules.sd_models

#         modules.sd_models.model_data.set_sd_model(value)


# sys.modules['modules.shared'].__class__ = Shared   

# (3) The effect of (2) is to define a property sd_model in the  module "modules.shared".
# In other words, the field "

# (4) It implies that the field "sd_model" of the module "modules.shared" becomes a property
# whose value is defined as modules.sd_models.model_data.get_sd_model().
# So whenever property "shared.sd_model" is mentioned somewhere, its value will be computed and
# the function modules.sd_models.model_data.get_sd_model() is called.

# (5) modules.sd_models.model_data.get_sd_model()  is defined as follows:

# class SdModelData:
#     def __init__(self):
#         self.sd_model = None
#         self.loaded_sd_models = []
#         self.was_loaded_at_least_once = False
#         self.lock = threading.Lock()

#     def get_sd_model(self): #MJ: called by property shared.sd_model in load_model() # noqa: B018 from
#         # def load_model(): #MJ: different from sd_models.load_model; The former is called by the thread
#         # """
#         # Accesses shared.sd_model property to load model.
#         # After it's available, if it has been loaded before this access by some extension,
#         # its optimization may be None because the list of optimizaers has neet been filled
#         # by that time, so we apply optimization again.
#         # """

#         # shared.sd_model  # noqa: B018

#         # if sd_hijack.current_optimizer is None:
#         #     sd_hijack.apply_optimizations()

#         # from modules import devicesion()

#         if self.was_loaded_at_least_once:
#             return self.sd_model

#         if self.sd_model is None:
            
#             with self.lock:
                
#                 if self.sd_model is not None or self.was_loaded_at_least_once:
#                     return self.sd_model

#                 try:
#                     load_model() #MJ:  def load_model(checkpoint_info=None, already_loaded_state_dict=None)
#                                   # =>  print(f"Creating model from config: {checkpoint_config}") #MJ: shared.sd_model = LatentDiffusion is done here
#                                   # This print statement in the thread is not encountered in the debugging process???

#                 except Exception as e:
#                     errors.display(e, "loading stable diffusion model", full_traceback=True)
#                     print("", file=sys.stderr)
#                     print("Stable diffusion model failed to load", file=sys.stderr)
#                     self.sd_model = None

#         return self.sd_model

#     def set_sd_model(self, v, already_loaded=False):
#         self.sd_model = v
#         if already_loaded:
#             sd_vae.base_vae = getattr(v, "base_vae", None)
#             sd_vae.loaded_vae_file = getattr(v, "loaded_vae_file", None)
#             sd_vae.checkpoint_info = v.sd_checkpoint_info

#         try:
#             self.loaded_sd_models.remove(v)
#         except ValueError:
#             pass

#         if v is not None:
#             self.loaded_sd_models.insert(0, v)


# model_data = SdModelData() #MJ: create a field "model_data" of module "modules.sd_models"

# (6) load_model() is defined as follows:

# def load_model(checkpoint_info=None, already_loaded_state_dict=None):
#     from modules import sd_hijack
#     checkpoint_info = checkpoint_info or select_checkpoint()

#     timer = Timer()

#     if model_data.sd_model:
#         send_model_to_trash(model_data.sd_model)
#         model_data.sd_model = None
#         devices.torch_gc()

#     timer.record("unload existing model")

#     if already_loaded_state_dict is not None:
#         state_dict = already_loaded_state_dict
#     else:
#         state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

#     checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
#     clip_is_included_into_sd = any(x for x in [sd1_clip_weight, sd2_clip_weight, sdxl_clip_weight, sdxl_refiner_clip_weight] if x in state_dict)

#     timer.record("find config")

#     sd_config = OmegaConf.load(checkpoint_config)  #MJ: load config
#     repair_config(sd_config)

#     timer.record("load config")

#     print(f"Creating model from config: {checkpoint_config}") #MJ: shared.sd_model = LatentDiffusion is done here

#     sd_model = None
#     try:
#         with sd_disable_initialization.DisableInitialization(disable_clip=clip_is_included_into_sd or shared.cmd_opts.do_not_download_clip):
#             with sd_disable_initialization.InitializeOnMeta():
#                 sd_model = instantiate_from_config(sd_config.model)

#     except Exception as e:
#         errors.display(e, "creating model quickly", full_traceback=True)

#     if sd_model is None:
#         print('Failed to create model quickly; will retry using slow method.', file=sys.stderr)

#         with sd_disable_initialization.InitializeOnMeta():
#             sd_model = instantiate_from_config(sd_config.model)

#     sd_model.used_config = checkpoint_config

#     timer.record("create model")

#     if shared.cmd_opts.no_half:
#         weight_dtype_conversion = None
#     else:
#         weight_dtype_conversion = {
#             'first_stage_model': None,
#             '': torch.float16,
#         }

#     with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=model_target_device(sd_model), weight_dtype_conversion=weight_dtype_conversion):
#         load_model_weights(sd_model, checkpoint_info, state_dict, timer)
#     timer.record("load weights from state dict")

#     send_model_to_device(sd_model)
#     timer.record("move model to device")

#     sd_hijack.model_hijack.hijack(sd_model)

#     timer.record("hijack")

#     sd_model.eval()
#     model_data.set_sd_model(sd_model)
#     model_data.was_loaded_at_least_once = True

#     sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)  # Reload embeddings after model load as they may or may not fit the model

#     timer.record("load textual inversion embeddings")

#     script_callbacks.model_loaded_callback(sd_model)

#     timer.record("scripts callbacks")

#     with devices.autocast(), torch.no_grad():
#         sd_model.cond_stage_model_empty_prompt = get_empty_cond(sd_model)

#     timer.record("calculate empty prompt")

#     print(f"Model loaded in {timer.summary()}.")

#     return sd_model

# (7) In load_model(),

# the following statement actually loads an object of LatentDiffusion to sd_model:

#   sd_model = instantiate_from_config(sd_config.model)

# Then, the following statement
#  model_data.set_sd_model(sd_model)

# Finally, the following return statement

# return self.sd_model  #MJ: self.sd_model was  set by  model_data.set_sd_model(sd_model) in load_model() 
# returns sd_model  which will be a value of  property modules.shared.sd_model.
# => This is my analysis. What do you think?

# So. do you mean we can change the class to which an instance belongs after the instance has been created?
# ChatGPT
# Yes, in Python, it is possible to change the class of an instance after it has been created, a technique often referred to as "monkey patching". This is done by altering the __class__ attribute of the instance to point to a different class.

# Here's a simple demonstration using Python classes and instances:

# python
# Copy code
# class A:
#     def method(self):
#         print("Method in class A")

# class B:
#     def method(self):
#         print("Method in class B")

# # Creating an instance of class A
# a_instance = A()

# # Calling the method of class A
# a_instance.method()  # Output: "Method in class A"

# # Changing the class of a_instance to B
# a_instance.__class__ = B

# # Now calling the method will call the method of class B
# a_instance.method()  # Output: "Method in class B"
# In this code:

# We first define two classes, A and B, each with a method that prints a different message.
# We create an instance of class A.
# We call the method of class A using the instance we created, which prints a message.
# We then change the class of the instance to B using the __class__ attribute.
# Now, when we call the method using the instance, it calls the method of class B, printing a different message.
# This example demonstrates how you can change the behavior of an instance dynamically by changing its class. However, you should be very careful when using this technique, as it can lead to code that is hard to understand and maintain. It is generally recommended to use other design patterns, like composition or inheritance, to achieve the desired behavior in a more structured way.




# User
# To use this technique both methods of class A and B should be the same?
# ChatGPT
# No, the methods do not necessarily need to have the same name or signature between the two classes. However, if you are changing the __class__ attribute of an instance and then calling a method that only exists in the original class, you will encounter an AttributeError, because the method does not exist in the new class.

# Here's an illustration:

# python
# Copy code
# class A:
#     def method_in_a(self):
#         print("Method in class A")

# class B:
#     def method_in_b(self):
#         print("Method in class B")

# # Creating an instance of class A
# a_instance = A()

# # Calling the method of class A
# a_instance.method_in_a()  # Output: "Method in class A"

# # Changing the class of a_instance to B
# a_instance.__class__ = B

# # Trying to call the method of class A will now raise an AttributeError
# try:
#     a_instance.method_in_a()
# except AttributeError as e:
#     print(f"AttributeError: {e}")  # Output: "AttributeError: 'B' object has no attribute 'method_in_a'"

# # But we can call the method of class B
# a_instance.method_in_b()  # Output: "Method in class B"
# In this example:

# Classes A and B have different methods: method_in_a and method_in_b, respectively.
# After changing the class of a_instance from A to B, calling method_in_a (a method of class A) raises an AttributeError.
# However, we can call method_in_b (a method of class B) on a_instance after changing its class to B.
# So while the methods do not need to have the same name, you need to ensure that you are calling methods that exist on the new class of the instance after changing its __class__ attribute to avoid errors.


