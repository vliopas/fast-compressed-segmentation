workspace "CSVEncoder"
    architecture "x64"
    startproject "CSVEncoder"
    toolset "v143"

    configurations
    {
        "Debug",
        "Release"
    }

    OutputDir = "%{prj.name}-%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
    OutputName ="%{prj.name}"

project "CSVEncoder"
    location "CSVEncoder"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++latest"
    staticruntime "off"
    floatingpoint "fast"

    targetdir ("bin/")
    objdir ("bin-int/".. OutputDir)

    includedirs
    {
        "%{prj.name}/src",
        "%{prj.name}/src/cnpy",
        "%{prj.name}/src/zlib/include"
    }

    libdirs
    {
        "%{prj.name}/src/zlib/lib"
    }

    links   
    { 
        "zlib" 
    } 

    files
    {
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.hpp",
        "%{prj.name}/src/**.cpp"
    }