node {
    docker.withRegistry('https://newknowledge.azurecr.io', 'acr-creds') {
    
        git url: "https://github.com/NewKnowledge/nk_croc.git", credentialsId: '055e98d5-ce0c-45ef-bf0d-ddc6ed9b634a'
    
        sh "git rev-parse HEAD > .git/commit-id"
        def commit_id = readFile('.git/commit-id').trim()
        println commit_id
    
        stage "build_docker_image"
        def http_image = docker.build("ds/croc-http", "-f http.dockerfile .")
    
        stage "publish_docker_image"
        def images = [http_image]
        def branches = sh(returnStdout: true, script: "git branch --contains ${commit_id}")
        for (image in images) {
            image.push "${BRANCH_NAME}"
            image.push "${commit_id}"
            if ("${BRANCH_NAME}" == "master") {
                image.push 'latest'
            }
        }
    }
}
