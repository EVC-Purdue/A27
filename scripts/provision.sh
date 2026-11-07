#!/usr/bin/env bash
# https://github.com/EVC-Purdue/A27/issues/15
# Set up a host for one role (dev, jetson, rubik or unoq); safe to rerun.
# shellcheck disable=SC1090,SC1091
set -euo pipefail

# settings

ROS_DISTRO="jazzy"
UBUNTU="noble" # Jazzy ships binaries for Ubuntu 24.04 only
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROS_KEY_URL="https://raw.githubusercontent.com/ros/rosdistro/master/ros.key"
ROS_KEYRING="/usr/share/keyrings/ros-archive-keyring.gpg"
ROS_APT_LIST="/etc/apt/sources.list.d/ros2.list"

ROS_PACKAGES=(
  "ros-$ROS_DISTRO-ros-base"
  "ros-$ROS_DISTRO-rmw-cyclonedds-cpp"
  "ros-$ROS_DISTRO-rosbag2-storage-mcap"
  python3-colcon-common-extensions
  python3-rosdep
  python3-pip
  python3-pytest
  build-essential
  git
)

# helpers

step() {
  echo
  echo "==> $*"
}

fail() {
  echo "error: $*" >&2
  exit 1
}

# Runs a command as root, through sudo unless we already are root (as in CI)
as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

# ROS's setup.bash reads unset variables, so -u is off while it loads
source_ros() {
  set +u
  . "/opt/ros/$ROS_DISTRO/setup.bash"
  set -u
}

# steps

check_os() {
  step "Check OS"
  [ -f /etc/os-release ] || fail "ROS 2 $ROS_DISTRO needs Ubuntu $UBUNTU, this is not Linux"
  . /etc/os-release
  echo "$PRETTY_NAME, workspace $WORKSPACE"
  [ "$VERSION_CODENAME" = "$UBUNTU" ] || fail "ROS 2 $ROS_DISTRO needs Ubuntu $UBUNTU, this is $VERSION_CODENAME"
}

add_ros_apt_source() {
  step "Add the ROS 2 apt source"
  if grep -Rqs "packages.ros.org/ros2" /etc/apt/sources.list /etc/apt/sources.list.d/; then
    echo "already added"
    return
  fi
  as_root apt-get update
  as_root apt-get install -y --no-install-recommends software-properties-common curl gnupg ca-certificates
  as_root add-apt-repository -y universe
  as_root curl -fsSL "$ROS_KEY_URL" -o "$ROS_KEYRING"

  local arch
  arch="$(dpkg --print-architecture)"
  echo "deb [arch=$arch signed-by=$ROS_KEYRING] http://packages.ros.org/ros2/ubuntu $UBUNTU main" \
    | as_root tee "$ROS_APT_LIST" > /dev/null
}

install_ros() {
  step "Install ROS 2 $ROS_DISTRO"
  as_root apt-get update
  as_root apt-get install -y --no-install-recommends "${ROS_PACKAGES[@]}"
}

install_workspace_deps() {
  step "Install workspace dependencies"
  if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    as_root rosdep init
  fi
  rosdep update
  source_ros
  rosdep install --from-paths "$WORKSPACE/src" --ignore-src -y --rosdistro "$ROS_DISTRO"
}

build_workspace() {
  step "Build the workspace"
  source_ros
  cd "$WORKSPACE"
  colcon build --symlink-install
  mkdir -p data logs figures
}

# roles

provision_dev() {
  check_os
  add_ros_apt_source
  install_ros
  install_workspace_deps
  build_workspace

  step "Done. Next:"
  echo "source $WORKSPACE/install/setup.bash"
  echo "ros2 launch autonomous_kart bringup_sim.launch.py"
}

main() {
  [ $# -eq 1 ] || fail "usage: $0 {dev|jetson|rubik|unoq}"
  case "$1" in
    dev) provision_dev ;;
    jetson | rubik | unoq) fail "role $1 is not supported yet (#15)" ;;
    *) fail "unknown role $1, expected dev, jetson, rubik or unoq" ;;
  esac
}

main "$@"
