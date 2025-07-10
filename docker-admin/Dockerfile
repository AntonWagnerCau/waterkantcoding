FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl bash sudo git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Phosphobot via official install.sh script
RUN curl -fsSL https://raw.githubusercontent.com/phospho-app/phosphobot/main/install.sh | bash

# Expose port used by Phosphobot (default is 80)
EXPOSE 80

# Start the Phosphobot service
CMD ["phosphobot", "run"]