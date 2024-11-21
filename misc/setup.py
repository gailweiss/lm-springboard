from model.transformer.transformer import Transformer
from misc.create import make_model
from misc.save_load import load_model, get_datamodule
from misc.model_explorer import get_full_path
from data.data_params import make_dp
from model.model_params import make_mp
from train.train_params import make_tp
from misc.util import printer_print as print


def sync_model_params(requested_model_params, loaded_model_params):
    # these factors are from the loaded model, so write them back:
    print("loaded a model and now syncing a subset of the model params!")
    print("if you have changed the model implementation - make sure")
    print("this function is syncing all the relevant params!")
    for a in ["n_layers", "n_heads", "dim", "dim_ff_factor",
              "tokenizer_source_name", "custom_tokenizer_ntokens",
              "layer_architecture", "from_os_pretrained",
              "individual_head_params", "pos_encoding", "max_seq_len"]:
        setattr(requested_model_params, a, getattr(loaded_model_params, a))


def setup_model_and_data(data_params, model_params, train_params, verbose=True,
                         skip_data=False, keep_datamodule=False):
    
    dataset = None

    loading = model_params.from_saved or model_params.from_os_pretrained    
    load_res = {}
    if loading:
        if model_params.from_saved:
            p = get_full_path(identifier, checkpoint=checkpoint)
            assert None is not p, f"didn't find path for identifier {identifier}"
            load_res = load_model(p, full=True, verbose=verbose, 
                                  with_data=not skip_data)
            dataset = load_res["dataset"]
        else:
            if model_params.from_os_pretrained == "gpt2":
                load_res["lm"] = get_gpt2()
            else:
                raise NotImplementedError("unknown pretrained model requested:" +
                                      f"{model_params.from_os_pretrained}")

        lm = load_res["lm"]
        sync_model_params(model_params, lm.model_params)

    if (None is dataset) and not skip_data:
        # model params already been synced, can use them for the datamodule
        dataset = get_datamodule(data_params, model_params, verbose=verbose,
            keep_datamodule=keep_datamodule)
        
    if not loading:  # ie, making
        assert not skip_data  # need data to determine the tokenizer
        lm = make_model(model_params, train_params, dataset.tokenizer)

    return lm, dataset


def quick_data_grab(dataset_name, tokenizer_source_name="gpt2", verbose=False, max_seq_len=200):
    dp = make_dp(dataset_name=dataset_name, debug_crop=500)
    mp = make_mp(tokenizer_source_name=tokenizer_source_name, max_seq_len=max_seq_len)
    return get_datamodule(dp, mp, verbose=verbose, keep_datamodule=False)
