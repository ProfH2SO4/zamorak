from zamorak.main import main

# find how NN project is loading config
# NN project must have params which he want to change PARAMS_TO_OPT = ["loss": {(min, max),  "config_tag": "LOSS"}]
# RUN NN project and parse log file
# Load right params, do Bayesian optimization and change the NN config


if __name__ == "__main__":
    main()